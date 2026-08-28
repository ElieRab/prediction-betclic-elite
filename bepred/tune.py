"""Reglage des hyperparametres par validation glissante.

Recherche coordonnee : chaque parametre est balaye a son tour, les autres
etant figes, jusqu'a stabilisation. La log-perte hors echantillon sert de
critere -- elle sanctionne la sur-confiance, ce que la simple justesse ignore.

Les ecarts observes entre reglages voisins sont faibles au regard du nombre
de matchs disponibles : le resultat est un reglage raisonnable, pas un
optimum a defendre au millieme.
"""
from __future__ import annotations

import copy

import pandas as pd

from .backtest import metrics, walk_forward
from .config import Config

#: (chemin dans la config, valeurs testees)
GRID: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("elo.k", (12.0, 18.0, 24.0, 30.0, 36.0)),
    ("elo.season_regress", (0.0, 0.15, 0.25, 0.40)),
    ("elo.new_team_penalty", (40.0, 70.0, 90.0, 120.0)),
    ("poisson.half_life_days", (120.0, 180.0, 240.0, 360.0, 500.0)),
    ("poisson.ridge", (2.0, 6.0, 12.0, 24.0)),
    ("blend.poisson_weight", (0.3, 0.5, 0.7, 0.85, 1.0)),
    ("blend.shrink_matches", (8.0, 20.0, 40.0)),
)


def _set(cfg: Config, path: str, value: float) -> None:
    group, name = path.split(".")
    setattr(getattr(cfg, group), name, value)


def _get(cfg: Config, path: str) -> float:
    group, name = path.split(".")
    return getattr(getattr(cfg, group), name)


def score(matches: pd.DataFrame, cfg: Config) -> dict:
    """Evalue une configuration en validation glissante."""
    frozen = copy.deepcopy(cfg)
    original = Config.load
    Config.load = staticmethod(lambda *a, **k: copy.deepcopy(frozen))
    try:
        return metrics(walk_forward(matches, frozen))
    finally:
        Config.load = original


def search(matches: pd.DataFrame, cfg: Config | None = None, rounds: int = 2,
           verbose: bool = True) -> tuple[Config, dict]:
    """Balaye la grille et renvoie la meilleure configuration trouvee."""
    cfg = copy.deepcopy(cfg or Config())
    best = score(matches, cfg)
    for _ in range(rounds):
        improved = False
        for path, values in GRID:
            current = _get(cfg, path)
            for value in values:
                if value == current:
                    continue
                trial = copy.deepcopy(cfg)
                _set(trial, path, value)
                got = score(matches, trial)
                if got["logloss"] < best["logloss"] - 1e-5:
                    best, cfg, current = got, trial, value
                    improved = True
                    if verbose:
                        print(f"  {path} = {value}  ->  log-perte {got['logloss']:.4f}")
        if not improved:
            break
    return cfg, best
