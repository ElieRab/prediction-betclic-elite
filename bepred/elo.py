"""Moteur Elo basket (marge de victoire, avantage du terrain, inter-saisons).

Formule usuelle du basket : le gain de points depend de la marge, amortie par
l'ecart de niveau (une equipe deja favorite gagne moins a s'imposer largement).
L'echelle est calibree pour qu'un ecart de `elo_per_point` points Elo vaille
un point d'ecart au score -- c'est le pont vers le modele Poisson.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import EloParams


@dataclass
class EloEngine:
    params: EloParams = field(default_factory=EloParams)
    ratings: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)   # une ligne par match
    n_matches: dict[str, int] = field(default_factory=dict)
    _last_season: int | None = None
    _last_seen: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------ internals
    def _init_rating(self) -> float:
        """Note d'arrivee d'un club jamais rencontre."""
        base = (float(np.mean(list(self.ratings.values()))) if self.ratings
                else self.params.start_rating)
        return base - self.params.new_team_penalty

    def rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self._init_rating()
            self.n_matches[team] = 0
        return self.ratings[team]

    def _season_break(self, season: int) -> None:
        """Regression vers la moyenne a chaque intersaison."""
        if self._last_season is None or season <= self._last_season:
            self._last_season = season
            return
        if self.ratings:
            mean = float(np.mean(list(self.ratings.values())))
            r = 1.0 - (1.0 - self.params.season_regress) ** (season - self._last_season)
            for t in self.ratings:
                self.ratings[t] = mean + (1.0 - r) * (self.ratings[t] - mean)
        self._last_season = season

    def _rust(self, team: str, season: int) -> None:
        """Un club absent du championnat revient plus proche de la moyenne.

        Faute de le voir jouer, on ne sait rien de son evolution : le laisser
        a sa note d'il y a deux ans reviendrait a surestimer notre information.
        """
        seen = self._last_seen.get(team)
        if seen is None or season - seen <= 1 or not self.ratings:
            return
        mean = float(np.mean(list(self.ratings.values())))
        r = 1.0 - (1.0 - self.params.season_regress) ** (season - seen - 1)
        self.ratings[team] = mean + (1.0 - r) * (self.ratings[team] - mean)
        self.ratings[team] -= self.params.promotion_shift
        self._last_seen[team] = season      # la decote ne s'applique qu'une fois

    # ---------------------------------------------------------------- public
    def expected(self, rh: float, ra: float) -> float:
        """Probabilite de victoire a domicile, avantage du terrain inclus."""
        return 1.0 / (1.0 + 10.0 ** (-(rh + self.params.home_adv - ra) / 400.0))

    def expected_margin(self, home: str, away: str) -> float:
        """Ecart de points attendu (positif = avantage a domicile)."""
        diff = (self.rating(home) + self.params.home_adv - self.rating(away))
        return diff / self.params.elo_per_point

    def mov_multiplier(self, margin: int, elo_diff_winner: float) -> float:
        """Multiplicateur de marge, amorti par l'ecart de niveau du vainqueur."""
        m = max(1.0, abs(float(margin)))
        p = self.params
        return (m + 3.0) ** p.mov_a / (p.mov_b + 0.006 * max(0.0, elo_diff_winner))

    def update(self, home: str, away: str, hp: int, ap: int, ot: int = 0,
               season: int | None = None, date=None, record: bool = True):
        if season is not None:
            self._season_break(season)
        for team in (home, away):
            self.rating(team)
            if season is not None:
                self._rust(team, season)
        rh, ra = self.ratings[home], self.ratings[away]

        exp_h = self.expected(rh, ra)
        margin = hp - ap
        # Un match gagne en prolongation vaut un succes d'un point : la
        # prolongation dit que les deux equipes etaient a egalite a la sirene.
        eff_margin = 1 if ot else margin
        s_h = 1.0 if margin > 0 else 0.0
        diff_winner = (rh + self.params.home_adv - ra) * (1 if margin > 0 else -1)
        delta = (self.params.k * self.mov_multiplier(eff_margin, diff_winner)
                 * (s_h - exp_h))

        self.ratings[home] = rh + delta
        self.ratings[away] = ra - delta
        self.n_matches[home] = self.n_matches.get(home, 0) + 1
        self.n_matches[away] = self.n_matches.get(away, 0) + 1
        if season is not None:
            self._last_seen[home] = self._last_seen[away] = season

        if record:
            self.history.append({
                "date": date, "season": season, "home": home, "away": away,
                "hp": hp, "ap": ap, "ot": ot,
                "elo_h_pre": rh, "elo_a_pre": ra, "exp_h": exp_h,
                "elo_h_post": self.ratings[home], "elo_a_post": self.ratings[away],
                "delta": delta,
            })
        return self.ratings[home], self.ratings[away]

    def recenter(self) -> None:
        """Ramene la moyenne du pool a 1500 : seuls les ecarts comptent."""
        if not self.ratings:
            return
        shift = self.params.start_rating - float(np.mean(list(self.ratings.values())))
        for team in self.ratings:
            self.ratings[team] += shift

    def run(self, matches: pd.DataFrame, recenter: bool = True) -> "EloEngine":
        """Rejoue tout l'historique dans l'ordre chronologique."""
        for m in matches.sort_values("date", kind="stable").itertuples(index=False):
            self.update(m.home, m.away, int(m.hp), int(m.ap), int(m.ot),
                        season=int(m.season), date=m.date)
        if recenter:
            self.recenter()
        return self

    # -------------------------------------------------------------- exports
    def table(self, teams: list[str] | None = None) -> pd.DataFrame:
        teams = teams or list(self.ratings)
        rows = [{"team": t, "elo": self.ratings.get(t, self.params.start_rating),
                 "matches": self.n_matches.get(t, 0)} for t in teams]
        return (pd.DataFrame(rows).sort_values("elo", ascending=False)
                .reset_index(drop=True))

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def trajectory(self, team: str) -> pd.DataFrame:
        """Courbe d'Elo d'un club (note apres chaque match)."""
        h = self.history_frame()
        if h.empty:
            return pd.DataFrame(columns=["date", "elo", "opponent", "venue"])
        mh, ma = h.home == team, h.away == team
        rows = pd.concat([
            pd.DataFrame({"date": h.loc[mh, "date"], "elo": h.loc[mh, "elo_h_post"],
                          "opponent": h.loc[mh, "away"], "venue": "D"}),
            pd.DataFrame({"date": h.loc[ma, "date"], "elo": h.loc[ma, "elo_a_post"],
                          "opponent": h.loc[ma, "home"], "venue": "E"}),
        ])
        return rows.sort_values("date").reset_index(drop=True)


def fit_elo_per_point(history: pd.DataFrame, params: EloParams) -> float:
    """Regresse l'ecart de points sur l'ecart Elo -> points Elo par point marque."""
    if len(history) < 200:
        return params.elo_per_point
    x = (history.elo_h_pre + params.home_adv - history.elo_a_pre).to_numpy(float)
    y = (history.hp - history.ap).to_numpy(float)
    beta = float(np.dot(x, y) / np.dot(x, x))       # points marques par point Elo
    if beta <= 1e-6:
        return params.elo_per_point
    return float(np.clip(1.0 / beta, 15.0, 60.0))


def fit_home_advantage(matches: pd.DataFrame, params: EloParams) -> float:
    """Avantage du terrain en points Elo, deduit de l'ecart moyen observe."""
    if matches.empty:
        return params.home_adv
    edge = float((matches.hp - matches.ap).mean())
    return float(np.clip(edge * params.elo_per_point, 0.0, 250.0))
