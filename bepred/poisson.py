"""Modele de points de type Poisson, adapte au basket.

    lambda_dom = exp(mu + gamma + attaque_dom - defense_ext)
    lambda_ext = exp(mu         + attaque_ext - defense_dom)

Les forces sont estimees par maximum de vraisemblance Poisson pondere
(decroissance exponentielle du poids des matchs anciens, regularisation L2),
avec gradient analytique.

Un Poisson pur predirait un ecart-type de sqrt(85) ~ 9 points par equipe, la
ou la Betclic Elite en affiche 12,4 : le score d'un match de basket est
sur-disperse, et les deux scores d'une meme rencontre sont correles (un match
a haut rythme fait monter les deux). Le modele ajoute donc deux ingredients :

* un facteur de rythme z partage par les deux equipes, z ~ Gamma(moyenne 1,
  variance `pace_var`), qui cree la correlation ;
* une dispersion residuelle propre a chaque equipe (binomiale negative de
  parametre `shape`), qui elargit chaque marginale.

Quand les deux valent zero (resp. l'infini), on retombe exactement sur deux
Poisson independants.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma as sp_gamma, nbinom, poisson as sp_poisson

from .config import PoissonParams

_N_NODES = 24
_MIN_PACE_VAR = 1e-5


# ------------------------------------------------------------------ melange
def pace_nodes(pace_var: float, n: int = _N_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Discretise le facteur de rythme z ~ Gamma(moyenne 1, variance `pace_var`).

    Stratification par quantiles equiprobables plutot que quadrature de
    Gauss-Laguerre : celle-ci perd toute precision numerique des que le
    parametre de forme depasse la centaine, ce qui est le cas ici.
    """
    if pace_var <= _MIN_PACE_VAR:
        return np.array([1.0]), np.array([1.0])
    shape = 1.0 / pace_var
    q = (np.arange(n) + 0.5) / n
    z = sp_gamma.ppf(q, shape, scale=pace_var)
    z = z / z.mean()                       # moyenne exactement 1
    return z, np.full(n, 1.0 / n)


def _marginal(mean: np.ndarray, shape: float, k: np.ndarray) -> np.ndarray:
    """Loi du score d'une equipe : Poisson si shape est infini, sinon NB."""
    if not np.isfinite(shape) or shape > 1e6:
        return sp_poisson.pmf(k[:, None], mean[None, :])
    p = shape / (shape + mean)
    return nbinom.pmf(k[:, None], shape, p[None, :])


@dataclass
class PointsModel:
    """Resultat d'un ajustement : forces d'attaque et de defense par club."""
    mu: float
    gamma: float
    attack: dict = field(default_factory=dict)
    defence: dict = field(default_factory=dict)
    weight: dict = field(default_factory=dict)
    pace_var: float = 0.0
    shape: float = float("inf")
    n_obs: int = 0
    total_weight: float = 0.0
    level_ratio: float = 1.0
    asof: object = None

    def has(self, team: str) -> bool:
        return team in self.attack

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        ah, dh = self.attack.get(home, 0.0), self.defence.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defence.get(away, 0.0)
        return (float(np.exp(self.mu + self.gamma + ah - da)),
                float(np.exp(self.mu + aa - dh)))

    @property
    def home_advantage_points(self) -> float:
        """Avantage du terrain exprime en points marques."""
        base = float(np.exp(self.mu))
        return base * (float(np.exp(self.gamma)) - 1.0)

    def strength_table(self, teams: list[str] | None = None) -> pd.DataFrame:
        teams = teams or sorted(self.attack)
        base = float(np.exp(self.mu))
        rows = []
        for t in teams:
            a, d = self.attack.get(t, 0.0), self.defence.get(t, 0.0)
            rows.append({
                "team": t,
                "attaque": base * float(np.exp(a)),
                "defense": base * float(np.exp(-d)),
                "attack_log": a, "defence_log": d,
                "poids": self.weight.get(t, 0.0),
            })
        df = pd.DataFrame(rows)
        df["net"] = df.attaque - df.defense
        return df.sort_values("net", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------- lois
    def score_matrix(self, lh: float, la: float, max_points: int = 140) -> np.ndarray:
        """Loi jointe M[i, j] = P(dom marque i, ext marque j)."""
        k = np.arange(max_points + 1)
        z, w = pace_nodes(self.pace_var)
        mat = np.zeros((max_points + 1, max_points + 1))
        ph = _marginal(lh * z, self.shape, k)      # (K+1, n_nodes)
        pa = _marginal(la * z, self.shape, k)
        for i, wi in enumerate(w):
            mat += wi * np.outer(ph[:, i], pa[:, i])
        total = mat.sum()
        return mat / total if total > 0 else mat

    def outcome(self, lh: np.ndarray, la: np.ndarray,
                max_points: int = 140) -> tuple[np.ndarray, np.ndarray]:
        """P(victoire dom en 40 min) et P(egalite), vectorise sur les matchs.

        Passe par les fonctions de repartition plutot que par la matrice
        complete : c'est ce qui rend la projection de saison abordable.
        """
        lh = np.atleast_1d(np.asarray(lh, float))
        la = np.atleast_1d(np.asarray(la, float))
        k = np.arange(max_points + 1)
        z, w = pace_nodes(self.pace_var)
        pw = np.zeros(lh.shape)
        pt = np.zeros(lh.shape)
        for zi, wi in zip(z, w):
            ph = _marginal(lh * zi, self.shape, k)     # (K+1, n_matchs)
            pa = _marginal(la * zi, self.shape, k)
            cum_a = np.cumsum(pa, axis=0)
            below = np.vstack([np.zeros((1, pa.shape[1])), cum_a[:-1]])
            pw += wi * (ph * below).sum(axis=0)
            pt += wi * (ph * pa).sum(axis=0)
        return pw, pt

    def sample(self, lh: np.ndarray, la: np.ndarray,
               rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Tirage de scores (rythme partage puis dispersion propre)."""
        lh = np.asarray(lh, float)
        la = np.asarray(la, float)
        if self.pace_var > _MIN_PACE_VAR:
            shape = 1.0 / self.pace_var
            z = rng.gamma(shape, self.pace_var, size=lh.shape)
        else:
            z = np.ones_like(lh)
        mh, ma = lh * z, la * z
        if np.isfinite(self.shape) and self.shape < 1e6:
            mh = rng.gamma(self.shape, mh / self.shape)
            ma = rng.gamma(self.shape, ma / self.shape)
        return rng.poisson(mh), rng.poisson(ma)


# ------------------------------------------------------------- ajustement
def _negll_and_grad(theta, idx_h, idx_a, yh, ya, w, n_teams, ridge):
    mu, gamma = theta[0], theta[1]
    atk = theta[2:2 + n_teams]
    dfc = theta[2 + n_teams:2 + 2 * n_teams]

    eta_h = mu + gamma + atk[idx_h] - dfc[idx_a]
    eta_a = mu + atk[idx_a] - dfc[idx_h]
    lh = np.exp(np.clip(eta_h, 2.0, 6.0))
    la = np.exp(np.clip(eta_a, 2.0, 6.0))

    nll = float(np.sum(w * (lh - yh * eta_h + la - ya * eta_a)))
    nll += ridge * float(np.sum(atk ** 2) + np.sum(dfc ** 2))

    rh = w * (lh - yh)        # residus ponderes
    ra = w * (la - ya)
    g = np.zeros_like(theta)
    g[0] = rh.sum() + ra.sum()
    g[1] = rh.sum()
    g[2:2 + n_teams] = (np.bincount(idx_h, rh, n_teams)
                        + np.bincount(idx_a, ra, n_teams) + 2 * ridge * atk)
    g[2 + n_teams:2 + 2 * n_teams] = (-np.bincount(idx_a, rh, n_teams)
                                      - np.bincount(idx_h, ra, n_teams)
                                      + 2 * ridge * dfc)
    return nll, g


def fit(matches: pd.DataFrame, params: PoissonParams | None = None,
        asof=None) -> PointsModel:
    """Ajuste les forces sur l'historique, matchs recents plus lourds."""
    params = params or PoissonParams()
    if matches.empty:
        return PointsModel(mu=float(np.log(83.0)), gamma=0.0)

    df = matches.sort_values("date", kind="stable")
    asof = asof or max(df.date)
    age = np.array([(asof - d).days for d in df.date], dtype=float)
    weights = 0.5 ** (np.maximum(age, 0.0) / params.half_life_days)

    teams = sorted(set(df.home) | set(df.away))
    index = {t: i for i, t in enumerate(teams)}
    idx_h = df.home.map(index).to_numpy(int)
    idx_a = df.away.map(index).to_numpy(int)
    yh = df.get("hp_reg", df.hp).to_numpy(float)
    ya = df.get("ap_reg", df.ap).to_numpy(float)

    n = len(teams)
    theta0 = np.zeros(2 + 2 * n)
    theta0[0] = np.log(max(1.0, float(np.average(np.r_[yh, ya],
                                                 weights=np.r_[weights, weights]))))
    res = minimize(_negll_and_grad, theta0, jac=True, method="L-BFGS-B",
                   args=(idx_h, idx_a, yh, ya, weights, n, params.ridge),
                   options={"maxiter": 800, "ftol": 1e-11})
    theta = res.x
    atk = theta[2:2 + n] - theta[2:2 + n].mean()
    dfc = theta[2 + n:2 + 2 * n] - theta[2 + n:2 + 2 * n].mean()
    mu = theta[0] + theta[2:2 + n].mean() - theta[2 + n:2 + 2 * n].mean()

    model = PointsModel(
        mu=float(mu), gamma=float(theta[1]),
        attack={t: float(atk[i]) for t, i in index.items()},
        defence={t: float(dfc[i]) for t, i in index.items()},
        weight={t: float(weights[(idx_h == i) | (idx_a == i)].sum())
                for t, i in index.items()},
        n_obs=int(len(df)), total_weight=float(weights.sum()), asof=asof,
    )
    _anchor_level(model, df, params.level_window)
    if params.fit_pace:
        model.pace_var, model.shape = estimate_dispersion(model, df, weights)
    else:
        model.pace_var, model.shape = params.pace_var, float("inf")
    return model


def _anchor_level(model: PointsModel, df: pd.DataFrame, window: int) -> None:
    """Recale le niveau de marque sur les matchs les plus recents.

    Le nombre de points marques en Betclic Elite monte vite (161 par match en
    2023-24, 173 en 2025-26). Une ponderation temporelle, meme courte, garde
    un pied dans le passe et sous-estime le total d'environ cinq points. Les
    forces relatives restent estimees sur tout l'historique ; seul le niveau
    d'ensemble est cale sur la derniere fenetre observee.
    """
    recent = df.tail(max(20, window))
    if recent.empty:
        return
    pred = np.array([sum(model.lambdas(h, a)) for h, a in zip(recent.home, recent.away)])
    obs = (recent.get("hp_reg", recent.hp).to_numpy(float)
           + recent.get("ap_reg", recent.ap).to_numpy(float))
    if pred.mean() <= 0:
        return
    model.mu += float(np.log(obs.mean() / pred.mean()))
    model.level_ratio = float(obs.mean() / pred.mean())


def estimate_dispersion(model: PointsModel, df: pd.DataFrame,
                        weights: np.ndarray) -> tuple[float, float]:
    """Deduit `pace_var` et `shape` des residus, par la methode des moments.

    Sous le modele : Var(Y) = m + m^2 (v + 1/shape) et Cov(dom, ext) = m_d m_e v.
    Deux moments, deux inconnues.
    """
    lam_h, lam_a = [], []
    for h, a in zip(df.home, df.away):
        lh, la = model.lambdas(h, a)
        lam_h.append(lh)
        lam_a.append(la)
    lam_h, lam_a = np.array(lam_h), np.array(lam_a)
    yh = df.get("hp_reg", df.hp).to_numpy(float)
    ya = df.get("ap_reg", df.ap).to_numpy(float)
    rh, ra = yh - lam_h, ya - lam_a
    w = weights / weights.sum()

    cov = float(np.sum(w * rh * ra))
    pace = cov / float(np.sum(w * lam_h * lam_a))
    pace = float(np.clip(pace, 0.0, 0.02))

    var = 0.5 * float(np.sum(w * (rh ** 2 + ra ** 2)))
    mean_lam = 0.5 * float(np.sum(w * (lam_h + lam_a)))
    mean_lam2 = 0.5 * float(np.sum(w * (lam_h ** 2 + lam_a ** 2)))
    extra = (var - mean_lam - pace * mean_lam2) / max(mean_lam2, 1e-9)
    shape = float("inf") if extra <= 1e-6 else float(1.0 / extra)
    return pace, shape
