"""Composition du championnat et projection de la saison.

Ce module a d'abord servi a projeter deux compositions concurrentes, le
seizieme club de la Betclic Elite 2026-2027 dependant d'une procedure
administrative. La decision est tombee : l'engagement de l'AS Monaco a ete
refuse, Saint-Quentin a ete repeche, et le championnat a demarre le
25 septembre 2026.

L'hypothese Monaco n'est plus projetee, et pas seulement parce qu'elle est
caduque : Saint-Quentin a joue, et rejouer la saison sans lui demanderait
d'effacer des resultats reels. La machinerie de comparaison reste en place --
elle resservira a la prochaine incertitude -- mais ne tourne qu'avec une
composition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import SCENARIOS, scenario_teams
from .season import project_season


def all_scenario_teams(matches: pd.DataFrame | None = None,
                       season: int | None = None) -> list[str]:
    """Union des clubs apparaissant dans au moins une composition."""
    teams: set[str] = set()
    for key in SCENARIOS:
        teams.update(composition(key, matches, season))
    return sorted(teams)


def composition(key: str, matches: pd.DataFrame | None = None,
                season: int | None = None) -> list[str]:
    """Les seize clubs d'une composition.

    Des que la saison a commence, la liste vient des resultats eux-memes :
    plus fiable qu'une liste tenue a la main, et automatiquement juste si un
    club change en cours de route.
    """
    declaree = scenario_teams(key)
    if matches is None or season is None or matches.empty:
        return declaree
    joues = matches[matches.season == season]
    vus = sorted(set(joues.home) | set(joues.away))
    return vus if len(vus) >= 2 else declaree


def run(model, matches: pd.DataFrame, cfg: Config, season: int,
        n_sims: int = 10000) -> dict[str, dict]:
    """Projette chaque scenario avec le meme modele et le meme alea."""
    model.prepare_season(season, all_scenario_teams(matches, season))
    out = {}
    for key, meta in SCENARIOS.items():
        teams = composition(key, matches, season)
        proj = project_season(model, matches, cfg, teams, season, n_sims=n_sims)
        proj["meta"] = meta
        proj["key"] = key
        out[key] = proj
    return out


def compare(runs: dict[str, dict], keys: tuple[str, ...] | None = None
            ) -> pd.DataFrame:
    """Ecart, club par club, entre les deux scenarios.

    Seuls les quinze clubs communs sont comparables ; Monaco et Saint-Quentin
    apparaissent avec la valeur du scenario ou ils jouent.
    """
    keys = keys or tuple(runs)
    if len(keys) < 2:
        return pd.DataFrame()       # une seule composition : rien a comparer
    a, b = runs[keys[0]], runs[keys[1]]
    fields = ("wins_mean", "p_top", "p_playin", "p_playoffs", "p_title", "p_rel")
    rows = []
    for team in sorted(set(a["teams"]) | set(b["teams"])):
        row: dict = {"club": team}
        for tag, run_ in (("a", a), ("b", b)):
            if team in run_["teams"]:
                i = run_["teams"].index(team)
                for f in fields:
                    row[f"{f}_{tag}"] = float(run_[f][i]) if f in run_ else np.nan
            else:
                for f in fields:
                    row[f"{f}_{tag}"] = np.nan
        for f in fields:
            row[f"{f}_d"] = row[f"{f}_a"] - row[f"{f}_b"]
        row["commun"] = team in a["teams"] and team in b["teams"]
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("wins_mean_a", ascending=False, na_position="last")


def headline(runs: dict[str, dict], key: str) -> dict:
    """Quelques reperes d'un scenario, pour l'entete de la page."""
    r = runs[key]
    i = int(np.argmax(r["p_title"]))
    j = int(np.argmax(r["p_rel"]))
    return {
        "favori": r["teams"][i], "p_favori": float(r["p_title"][i]),
        "menace": r["teams"][j], "p_menace": float(r["p_rel"][j]),
        "seizieme": r["meta"]["seizieme"],
    }
