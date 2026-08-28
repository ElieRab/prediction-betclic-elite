"""Projection d'une saison complete par simulation de Monte-Carlo.

On tire les scores de tous les matchs restants, on reconstitue le classement,
puis on deroule le play-in et les playoffs. Les series se resolvent
analytiquement (probabilite exacte de gagner un best-of-3 ou un best-of-5
selon l'alternance des terrains) : moins de bruit pour le meme cout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import round_robin

#: Alternance des terrains, du point de vue de la mieux classee.
BO3 = (True, False, True)
BO5 = (True, True, False, False, True)


# --------------------------------------------------------------- classement
def current_table(matches: pd.DataFrame, season: int,
                  teams: list[str] | None = None) -> pd.DataFrame:
    """Classement reel a date : victoires, defaites, points marques/encaisses."""
    df = matches[matches.season == season]
    teams = teams or sorted(set(df.home) | set(df.away))
    rows = {t: {"team": t, "j": 0, "v": 0, "d": 0, "pp": 0, "pc": 0} for t in teams}
    for m in df.itertuples(index=False):
        for team, own, opp in ((m.home, m.hp, m.ap), (m.away, m.ap, m.hp)):
            if team not in rows:
                continue
            r = rows[team]
            r["j"] += 1
            r["pp"] += int(own)
            r["pc"] += int(opp)
            r["v" if own > opp else "d"] += 1
    out = pd.DataFrame(rows.values())
    out["diff"] = out.pp - out.pc
    # Bareme LNB : 2 points par victoire, 1 par defaite.
    out["pts"] = 2 * out.v + out.d
    return (out.sort_values(["v", "diff"], ascending=False)
            .reset_index(drop=True))


def remaining_fixtures(matches: pd.DataFrame, season: int,
                       teams: list[str]) -> pd.DataFrame:
    """Affiches restantes : toutes les confrontations aller-retour non jouees.

    La Betclic Elite est un championnat aller-retour integral, ce qui suffit a
    deduire le programme restant du seul releve des matchs deja disputes.
    """
    played = {(m.home, m.away) for m in
              matches[matches.season == season].itertuples(index=False)}
    rows = []
    for i, day in enumerate(round_robin(sorted(teams)), start=1):
        for home, away in day:
            if (home, away) not in played:
                rows.append({"journee": i, "home": home, "away": away})
    return pd.DataFrame(rows, columns=["journee", "home", "away"])


# ------------------------------------------------------------------- series
def series_prob(a: np.ndarray, b: np.ndarray, pattern: tuple[bool, ...]) -> np.ndarray:
    """Probabilite que la mieux classee remporte la serie.

    `a` : proba de gagner un match chez elle, `b` : a l'exterieur.
    `pattern` : True quand la mieux classee recoit.
    """
    need = len(pattern) // 2 + 1
    shape = np.shape(a)
    # state[i, j] = proba d'etre a i victoires / j defaites.
    state = {(0, 0): np.ones(shape)}
    for at_home in pattern:
        p = a if at_home else b
        nxt: dict[tuple[int, int], np.ndarray] = {}
        for (w, l), prob in state.items():
            if w >= need or l >= need:
                nxt[(w, l)] = nxt.get((w, l), 0.0) + prob
                continue
            nxt[(w + 1, l)] = nxt.get((w + 1, l), 0.0) + prob * p
            nxt[(w, l + 1)] = nxt.get((w, l + 1), 0.0) + prob * (1.0 - p)
        state = nxt
    return sum(p for (w, _), p in state.items() if w >= need)


def pair_matrices(model, teams: list[str]) -> dict[str, np.ndarray]:
    """Matrices de confrontation entre tous les clubs : P(i bat j chez i)."""
    n = len(teams)
    home = np.repeat(teams, n)
    away = np.tile(teams, n)
    keep = home != away
    pred = model.predict_many(home[keep], away[keep])
    p = np.full(n * n, 0.5)
    p[keep] = pred.p_home.to_numpy()
    p = p.reshape(n, n)
    np.fill_diagonal(p, 0.5)
    a = p                       # i recoit j
    b = 1.0 - p.T               # i se deplace chez j
    return {"game": p, "bo3": series_prob(a, b, BO3), "bo5": series_prob(a, b, BO5)}


# --------------------------------------------------------------- projection
def project_season(model, matches: pd.DataFrame, cfg: Config, teams: list[str],
                   season: int, n_sims: int = 10000, seed: int = 12345) -> dict:
    """Simule la fin de saison et renvoie les probabilites par club."""
    rng = np.random.default_rng(seed)
    teams = sorted(teams)
    n = len(teams)
    index = {t: i for i, t in enumerate(teams)}

    table = current_table(matches, season, teams).set_index("team")
    wins0 = table.loc[teams, "v"].to_numpy(float)
    diff0 = table.loc[teams, "diff"].to_numpy(float)
    played0 = table.loc[teams, "j"].to_numpy(int)

    fx = remaining_fixtures(matches, season, teams)
    if fx.empty:
        wins = np.tile(wins0, (n_sims, 1))
        diff = np.tile(diff0, (n_sims, 1))
    else:
        pred = model.predict_many(fx.home.tolist(), fx.away.tolist())
        lh = pred.lam_h.to_numpy()
        la = pred.lam_a.to_numpy()
        edge = model.ot_edge(fx.home.tolist(), fx.away.tolist())
        ih = fx.home.map(index).to_numpy()
        ia = fx.away.map(index).to_numpy()

        m = len(fx)
        hp, ap = model.sample_scores(np.tile(lh, n_sims), np.tile(la, n_sims),
                                     np.tile(edge, n_sims), rng)
        hp = hp.reshape(n_sims, m)
        ap = ap.reshape(n_sims, m)
        home_win = hp > ap
        margin = (hp - ap).astype(float)

        wins = np.tile(wins0, (n_sims, 1))
        diff = np.tile(diff0, (n_sims, 1))
        for t in range(n):
            fh = np.flatnonzero(ih == t)
            fa = np.flatnonzero(ia == t)
            wins[:, t] += home_win[:, fh].sum(1) + (~home_win[:, fa]).sum(1)
            diff[:, t] += margin[:, fh].sum(1) - margin[:, fa].sum(1)

    # Classement : victoires, puis difference de points (le reglement passe
    # d'abord par les confrontations directes ; l'ecart general en est un
    # substitut fidele a l'echelle d'une distribution).
    score = wins * 10000.0 + diff + rng.random(wins.shape) * 1e-3
    order = np.argsort(-score, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n)[None, :], axis=1)

    rank_counts = np.zeros((n, n))
    for r in range(n):
        np.add.at(rank_counts[:, r], order[:, r], 1)
    rank_dist = rank_counts / n_sims

    out = {
        "teams": teams, "season": season, "n_sims": n_sims,
        "rank_dist": rank_dist,
        "wins_mean": wins.mean(0), "wins_p10": np.percentile(wins, 10, axis=0),
        "wins_p90": np.percentile(wins, 90, axis=0),
        "diff_mean": diff.mean(0),
        "played": played0, "wins_now": wins0, "diff_now": diff0,
        "fixtures": fx,
    }
    z = cfg.zones
    out["p_top"] = rank_dist[:, :z.direct].sum(1)
    out["p_playin"] = rank_dist[:, z.direct:z.playin].sum(1)
    out["p_rel"] = rank_dist[:, z.relegation - 1:].sum(1)
    out["p_first"] = rank_dist[:, 0]

    out.update(_postseason(model, teams, order, cfg, rng))
    return out


def _postseason(model, teams: list[str], order: np.ndarray, cfg: Config,
                rng: np.random.Generator) -> dict:
    """Deroule play-in puis playoffs a partir des tetes de serie simulees."""
    n = len(teams)
    n_sims = order.shape[0]
    z = cfg.zones
    if n < z.playin + 2:
        return {}
    mats = pair_matrices(model, teams)
    game, bo3, bo5 = mats["game"], mats["bo3"], mats["bo5"]

    def draw(hi: np.ndarray, lo: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Vainqueur d'une confrontation, la premiere equipe ayant l'avantage."""
        won = rng.random(hi.shape) < table[hi, lo]
        return np.where(won, hi, lo)

    s = order                                   # s[:, k] = club classe k+1
    # -- play-in : 7 recoit 8, 9 recoit 10, puis le perdant de 7-8 recoit
    #    le vainqueur de 9-10 pour la derniere place.
    g1w = draw(s[:, 6], s[:, 7], game)
    g1l = np.where(g1w == s[:, 6], s[:, 7], s[:, 6])
    g2w = draw(s[:, 8], s[:, 9], game)
    g3w = draw(g1l, g2w, game)

    seeds = np.column_stack([s[:, :z.direct], g1w, g3w])   # 8 qualifies
    p_playoffs = np.zeros(n)
    np.add.at(p_playoffs, seeds.ravel(), 1)
    p_playoffs /= n_sims

    # -- quarts (1-8, 2-7, 3-6, 4-5), demies, finale
    qf = [draw(seeds[:, i], seeds[:, 7 - i], bo3) for i in range(4)]
    sf1 = draw(*_reseed(qf[0], qf[3], seeds), bo3)
    sf2 = draw(*_reseed(qf[1], qf[2], seeds), bo3)
    champ = draw(*_reseed(sf1, sf2, seeds), bo5)

    p_final = np.zeros(n)
    np.add.at(p_final, np.concatenate([sf1, sf2]), 1)
    p_title = np.zeros(n)
    np.add.at(p_title, champ, 1)
    return {"p_playoffs": p_playoffs, "p_final": p_final / n_sims,
            "p_title": p_title / n_sims}


def _reseed(a: np.ndarray, b: np.ndarray, seeds: np.ndarray):
    """Remet la mieux classee des deux en position de recevoir."""
    n_sims, k = seeds.shape
    rows = np.arange(n_sims)
    pos = np.full((n_sims, int(seeds.max()) + 1), k, dtype=int)
    for j in range(k):                       # position de chaque club dans la grille
        pos[rows, seeds[:, j]] = j
    better = pos[rows, a] <= pos[rows, b]
    return np.where(better, a, b), np.where(better, b, a)


def fixture_predictions(model, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Previsions match par match pour un programme donne."""
    if fixtures.empty:
        return pd.DataFrame(columns=["journee", "home", "away", "p_home",
                                     "lam_h", "lam_a", "marge", "total"])
    pred = model.predict_many(fixtures.home.tolist(), fixtures.away.tolist())
    pred.insert(0, "journee", fixtures.journee.to_numpy())
    return pred
