"""Scenarios de composition de la Betclic Elite 2026-2027.

Quinze clubs sont certains d'y figurer. Le seizieme depend d'une procedure
administrative : soit l'AS Monaco obtient son engagement, soit Saint-Quentin,
relegue sportivement, est repeche. Le championnat n'etant pas le meme dans
les deux cas -- ni le seizieme club, ni donc les adversaires des quinze
autres -- chaque scenario est projete separement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .data import SCENARIOS, scenario_teams
from .season import project_season


def all_scenario_teams() -> list[str]:
    """Union des clubs apparaissant dans au moins un scenario."""
    teams: set[str] = set()
    for key in SCENARIOS:
        teams.update(scenario_teams(key))
    return sorted(teams)


def run(model, matches: pd.DataFrame, cfg: Config, season: int,
        n_sims: int = 10000) -> dict[str, dict]:
    """Projette chaque scenario avec le meme modele et le meme alea."""
    model.prepare_season(season, all_scenario_teams())
    out = {}
    for key, meta in SCENARIOS.items():
        teams = scenario_teams(key)
        proj = project_season(model, matches, cfg, teams, season, n_sims=n_sims)
        proj["meta"] = meta
        proj["key"] = key
        out[key] = proj
    return out


def compare(runs: dict[str, dict], keys: tuple[str, str] = ("monaco", "saint-quentin")
            ) -> pd.DataFrame:
    """Ecart, club par club, entre les deux scenarios.

    Seuls les quinze clubs communs sont comparables ; Monaco et Saint-Quentin
    apparaissent avec la valeur du scenario ou ils jouent.
    """
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
