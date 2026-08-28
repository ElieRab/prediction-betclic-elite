"""Assemblage Elo + Poisson.

L'Elo dit *de combien* une equipe est meilleure ; le modele de points dit
*comment* elle marque (rythme, attaque, defense). Les deux estimations de
l'ecart attendu sont melangees, puis retraduites en couple (lambda_dom,
lambda_ext) qui alimente la loi de score.

Le poids du Poisson est module par le volume de matchs recents disponibles
pour chaque club : un promu sans historique exploitable est juge presque
uniquement sur son Elo d'arrivee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .config import OT_MINUTES, REGULATION_MINUTES, Config
from .data import regulation_scores
from .elo import EloEngine, fit_elo_per_point, fit_home_advantage
from .poisson import PointsModel, fit as fit_points

_OT_SHARE = OT_MINUTES / REGULATION_MINUTES


@dataclass
class BEModel:
    cfg: Config
    elo: EloEngine
    points: PointsModel
    asof: date | None = None
    n_matches: int = 0
    _elo_iters: int = 0
    league_total: float = field(default=166.0)

    # ------------------------------------------------------------- build
    @classmethod
    def build(cls, matches: pd.DataFrame, cfg: Config | None = None) -> "BEModel":
        cfg = cfg or Config.load()
        reg = regulation_scores(matches)

        # L'avantage du terrain et l'echelle Elo -> points se determinent
        # l'un l'autre : quelques iterations suffisent a les stabiliser.
        for _ in range(4):
            hist = EloEngine(cfg.elo).run(matches).history_frame()
            if hist.empty:
                break
            cfg.elo.elo_per_point = fit_elo_per_point(hist, cfg.elo)
            cfg.elo.home_adv = fit_home_advantage(matches, cfg.elo)
        engine = EloEngine(cfg.elo).run(matches)

        points = fit_points(reg, cfg.poisson)
        # Niveau de reference pour les clubs sans historique : le total moyen
        # reellement observe sur la derniere fenetre, pas la valeur centrale
        # du modele (que l'inegalite de Jensen decale d'un point).
        window = reg.tail(max(20, cfg.poisson.level_window))
        total = (float((window.hp_reg + window.ap_reg).mean())
                 if len(window) else 166.0)
        return cls(cfg=cfg, elo=engine, points=points,
                   asof=(max(matches.date) if len(matches) else None),
                   n_matches=len(matches), _elo_iters=4, league_total=total)

    def prepare_season(self, season: int, teams: list[str]) -> None:
        """Amene les notes Elo au coup d'envoi d'une saison donnee.

        Applique la regression d'intersaison, puis, pour chaque club revenant
        d'une division inferieure ou jamais vu, la decote correspondante.
        """
        self.elo._season_break(season)
        for t in teams:
            self.elo.rating(t)
            self.elo._rust(t, season)

    # --------------------------------------------------------- confiance
    def confidence(self, team: str) -> float:
        """Part de credit accordee aux forces attaque/defense d'un club."""
        w = self.points.weight.get(team, 0.0)
        return float(w / (w + self.cfg.blend.shrink_matches))

    # ---------------------------------------------------------- lambdas
    def lambdas(self, home, away, neutral: bool = False):
        """Points attendus pour chaque equipe, apres melange Elo / Poisson."""
        home = np.atleast_1d(np.asarray(home, dtype=object))
        away = np.atleast_1d(np.asarray(away, dtype=object))
        b = self.cfg.blend

        lam_h, lam_a, m_elo, conf = [], [], [], []
        for h, a in zip(home, away):
            lh, la = self.points.lambdas(h, a)
            if neutral:
                half = 0.5 * self.points.home_advantage_points
                lh, la = lh - half, la + half
            lam_h.append(lh)
            lam_a.append(la)
            diff = self.elo.rating(h) - self.elo.rating(a)
            if not neutral:
                diff += self.cfg.elo.home_adv
            m_elo.append(diff / self.cfg.elo.elo_per_point)
            conf.append(min(self.confidence(h), self.confidence(a)))

        lam_h = np.array(lam_h, float)
        lam_a = np.array(lam_a, float)
        m_elo = np.array(m_elo, float)
        conf = np.array(conf, float)

        w = b.poisson_weight * conf
        margin = w * (lam_h - lam_a) + (1.0 - w) * m_elo
        # Le total revient au niveau de la ligue faute d'historique fiable.
        total = conf * (lam_h + lam_a) + (1.0 - conf) * self.league_total
        return 0.5 * (total + margin), 0.5 * (total - margin)

    # -------------------------------------------------------- previsions
    def ot_edge(self, home, away, neutral: bool = False) -> np.ndarray:
        """Probabilite que le club a domicile gagne une prolongation.

        Cinq minutes suffisent rarement a exprimer un ecart de niveau :
        l'avantage y est fortement attenue.
        """
        home = np.atleast_1d(np.asarray(home, dtype=object))
        away = np.atleast_1d(np.asarray(away, dtype=object))
        adv = 0.0 if neutral else self.cfg.elo.home_adv
        diff = np.array([self.elo.rating(h) + adv - self.elo.rating(a)
                         for h, a in zip(home, away)], dtype=float)
        p = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        return 0.5 + self.cfg.blend.ot_slope * (p - 0.5)

    def predict_many(self, home, away, neutral: bool = False) -> pd.DataFrame:
        lh, la = self.lambdas(home, away, neutral)
        pw, pt = self.points.outcome(lh, la)
        edge = self.ot_edge(home, away, neutral)
        p_home = pw + pt * edge
        return pd.DataFrame({
            "home": list(np.atleast_1d(home)), "away": list(np.atleast_1d(away)),
            "lam_h": lh, "lam_a": la,
            "marge": lh - la, "total": lh + la,
            "p_home": p_home, "p_away": 1.0 - p_home, "p_ot": pt,
        })

    def predict(self, home: str, away: str, neutral: bool = False) -> dict:
        row = self.predict_many([home], [away], neutral).iloc[0].to_dict()
        row["elo_h"] = self.elo.rating(home)
        row["elo_a"] = self.elo.rating(away)
        row["conf"] = min(self.confidence(home), self.confidence(away))
        return row

    def score_grid(self, home: str, away: str, neutral: bool = False,
                   span: int = 26) -> tuple[np.ndarray, int, int]:
        """Extrait de la loi jointe des scores autour des valeurs attendues."""
        lh, la = self.lambdas([home], [away], neutral)
        mat = self.points.score_matrix(float(lh[0]), float(la[0]),
                                       self.cfg.poisson.max_points)
        ch, ca = int(round(lh[0])), int(round(la[0]))
        h0 = max(0, ch - span // 2)
        a0 = max(0, ca - span // 2)
        return mat[h0:h0 + span, a0:a0 + span], h0, a0

    # ------------------------------------------------------------ tirage
    def sample_scores(self, lh: np.ndarray, la: np.ndarray, edge: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Tire des scores finaux, prolongations comprises."""
        hp, ap = self.points.sample(lh, la, rng)
        for _ in range(6):
            tie = hp == ap
            if not tie.any():
                break
            # Une prolongation de 5 minutes, au rythme du match, avec un
            # avantage attenue pour la meilleure equipe.
            base = _OT_SHARE * 0.5 * (lh[tie] + la[tie])
            tilt = 1.0 + 0.35 * (edge[tie] - 0.5)
            oh, oa = self.points.sample(base * tilt, base * (2.0 - tilt), rng)
            hp[tie] += oh
            ap[tie] += oa
        tie = hp == ap
        if tie.any():                       # garde-fou : aucun match nul ne sort
            hp[tie] += (rng.random(int(tie.sum())) < edge[tie]).astype(hp.dtype)
            ap[tie] += (hp[tie] == ap[tie]).astype(ap.dtype)
        return hp, ap
