"""Controle a posteriori : le modele est-il calibre ?

Validation glissante. Les matchs sont parcourus dans l'ordre ; par blocs, le
modele est reconstruit sur le seul passe puis note sur le bloc suivant. Aucun
resultat evalue n'a donc servi a l'ajustement.

Reperes : le score de Brier vaut 0,25 pour une piece equilibree ; predire
systematiquement l'equipe qui recoit donne le point de comparaison honnete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .model import BEModel

MIN_TRAIN = 240          # une saison complete avant la premiere evaluation


def walk_forward(matches: pd.DataFrame, cfg: Config | None = None,
                 step: int = 20) -> pd.DataFrame:
    """Previsions hors echantillon, match par match."""
    cfg = cfg or Config.load()
    df = matches.sort_values("date", kind="stable").reset_index(drop=True)
    rows = []
    for start in range(MIN_TRAIN, len(df), step):
        train = df.iloc[:start]
        test = df.iloc[start:start + step]
        model = BEModel.build(train, Config.load())
        pred = model.predict_many(test.home.tolist(), test.away.tolist())
        block = test.reset_index(drop=True)
        rows.append(pd.DataFrame({
            "date": block.date, "season": block.season,
            "home": block.home, "away": block.away,
            "hp": block.hp, "ap": block.ap,
            "p_home": pred.p_home.to_numpy(),
            "lam_h": pred.lam_h.to_numpy(), "lam_a": pred.lam_a.to_numpy(),
            "marge_prev": pred.marge.to_numpy(), "total_prev": pred.total.to_numpy(),
        }))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["gagne"] = (out.hp > out.ap).astype(int)
    out["marge"] = out.hp - out.ap
    out["total"] = out.hp + out.ap
    return out


def metrics(pred: pd.DataFrame) -> dict:
    """Brier, log-perte, justesse, erreurs de score -- et deux references."""
    if pred.empty:
        return {}
    p = pred.p_home.to_numpy(float)
    y = pred.gagne.to_numpy(float)
    eps = 1e-9
    base = float(y.mean())
    return {
        "n": int(len(pred)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "justesse": float(np.mean((p > 0.5) == (y > 0.5))),
        "brier_domicile": float(np.mean((base - y) ** 2)),
        "justesse_domicile": base,
        "mae_marge": float(np.mean(np.abs(pred.marge_prev - pred.marge))),
        "mae_total": float(np.mean(np.abs(pred.total_prev - pred.total))),
        "biais_marge": float(np.mean(pred.marge_prev - pred.marge)),
        "biais_total": float(np.mean(pred.total_prev - pred.total)),
    }


def reliability(pred: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Courbe de fiabilite : frequence observee par tranche de probabilite."""
    if pred.empty:
        return pd.DataFrame()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(pred.p_home, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "tranche": f"{100 * edges[b]:.0f}-{100 * edges[b + 1]:.0f} %",
            "n": int(m.sum()),
            "prevu": float(pred.p_home[m].mean()),
            "observe": float(pred.gagne[m].mean()),
        })
    return pd.DataFrame(rows)


def margin_check(pred: pd.DataFrame) -> dict:
    """Le modele se trompe-t-il d'ampleur ? (dispersion des ecarts)"""
    if pred.empty:
        return {}
    resid = (pred.marge - pred.marge_prev).to_numpy(float)
    return {
        "ecart_type_residu": float(np.std(resid)),
        "ecart_type_marge": float(np.std(pred.marge)),
        "correlation": float(np.corrcoef(pred.marge_prev, pred.marge)[0, 1]),
    }
