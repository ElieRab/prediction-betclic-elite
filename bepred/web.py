"""Interface web (Flask).

Le modele et les projections coutent quelques secondes : ils sont construits
une fois puis gardes en memoire. Le bouton "Actualiser" force un
retelechargement des resultats et un reajustement complet.
"""
from __future__ import annotations

import threading
from datetime import datetime

import numpy as np
from flask import Flask, redirect, render_template, request, url_for

from . import scenarios as sc
from .backtest import margin_check, metrics, reliability, walk_forward
from .config import Config, current_season, season_label
from .data import (SCENARIOS, display_name, load_matches, refresh,
                   resolve_team, slug, team_color)
from .model import BEModel
from .season import fixture_predictions

DEFAULT_SIMS = 10000
DEFAULT_SCENARIO = "monaco"

_lock = threading.Lock()
_state: dict = {}


def get_state(force: bool = False, n_sims: int = DEFAULT_SIMS) -> dict:
    """Construit (ou renvoie) le modele et les projections des deux scenarios."""
    with _lock:
        if _state and not force:
            return _state
        cfg = Config.load()
        report = refresh(cfg) if force else {}
        matches = load_matches(cfg, force=False)
        model = BEModel.build(matches, cfg)
        season = current_season()
        runs = sc.run(model, matches, cfg, season, n_sims=n_sims)
        played = matches[matches.season == season]
        _state.clear()
        _state.update({
            "cfg": cfg, "matches": matches, "model": model, "season": season,
            "runs": runs, "built_at": datetime.now(), "n_sims": n_sims,
            "last_result": (max(matches.date) if len(matches) else None),
            "joues": len(played), "report": report, "backtest": None,
        })
        return _state


def get_backtest() -> dict:
    """Validation glissante, calculee a la demande et mise en cache."""
    st = get_state()
    if st["backtest"] is None:
        pred = walk_forward(st["matches"], st["cfg"])
        st["backtest"] = {"pred": pred, "metrics": metrics(pred),
                          "reliability": reliability(pred), "marge": margin_check(pred)}
    return st["backtest"]


def pick_scenario(default: str = DEFAULT_SCENARIO) -> str:
    key = request.args.get("sc", default)
    return key if key in SCENARIOS else default


# ------------------------------------------------------------------ couleurs
def heat(value: float, warm: bool = True) -> str:
    """Fond degrade pour une probabilite (vert = favorable, rouge = defavorable)."""
    v = max(0.0, min(1.0, float(value)))
    if v < 0.005:
        return "transparent"
    hue = 145 if warm else 5
    alpha = 0.10 + 0.62 * v ** 0.75
    return f"hsla({hue}, 62%, 45%, {alpha:.3f})"


def _rows(proj: dict) -> list[dict]:
    """Une ligne par club, triee par nombre de victoires projete."""
    rows = []
    for i, team in enumerate(proj["teams"]):
        rows.append({
            "club": team,
            "joues": int(proj["played"][i]),
            "v_now": int(proj["wins_now"][i]),
            "diff_now": int(proj["diff_now"][i]),
            "v": float(proj["wins_mean"][i]),
            "v_lo": float(proj["wins_p10"][i]),
            "v_hi": float(proj["wins_p90"][i]),
            "diff": float(proj["diff_mean"][i]),
            "top": float(proj["p_top"][i]),
            "playin": float(proj["p_playin"][i]),
            "po": float(proj.get("p_playoffs", np.zeros(len(proj["teams"])))[i]),
            "finale": float(proj.get("p_final", np.zeros(len(proj["teams"])))[i]),
            "titre": float(proj.get("p_title", np.zeros(len(proj["teams"])))[i]),
            "premier": float(proj["p_first"][i]),
            "rel": float(proj["p_rel"][i]),
            "rangs": proj["rank_dist"][i],
        })
    rows.sort(key=lambda r: (-r["v"], -r["diff"]))
    for k, r in enumerate(rows, start=1):
        r["rang"] = k
    return rows


def create_app() -> Flask:
    app = Flask(__name__)
    app.jinja_env.filters["pct"] = lambda v: ("-" if v is None or v < 0.0005
                                              else f"{100 * v:.1f}")
    app.jinja_env.filters["pct0"] = lambda v: ("-" if v is None or v < 0.005
                                               else f"{100 * v:.0f}")
    app.jinja_env.filters["heat"] = heat
    app.jinja_env.filters["cold"] = lambda v: heat(v, warm=False)
    app.jinja_env.filters["nom"] = display_name
    app.jinja_env.filters["couleur"] = team_color
    app.jinja_env.filters["signe"] = lambda v: f"{v:+.0f}"
    app.jinja_env.globals["statique"] = False   # bascule a l'export
    app.jinja_env.filters["slug"] = slug

    @app.context_processor
    def inject():
        st = get_state()
        key = pick_scenario()
        return {
            "saison": season_label(st["season"]),
            "maj": st["last_result"],
            "construit": st["built_at"],
            "joues": st["joues"],
            "zones": st["cfg"].zones,
            "scenarios": SCENARIOS,
            "sc": key,
            "sc_meta": SCENARIOS[key],
            "nav": [("Projection", "index"), ("Scénarios", "scenarios_page"),
                    ("Rangs", "rangs"), ("Calendrier", "calendrier"),
                    ("Match", "match"), ("Forces", "forces"),
                    ("Fiabilité", "fiabilite")],
        }

    # ------------------------------------------------------------ projection
    @app.route("/")
    def index():
        st = get_state()
        key = pick_scenario()
        proj = st["runs"][key]
        rows = _rows(proj)
        return render_template("index.html", rows=rows, proj=proj,
                               n_sims=st["n_sims"],
                               favori=max(rows, key=lambda r: r["titre"]),
                               menace=max(rows, key=lambda r: r["rel"]),
                               restants=len(proj["fixtures"]))

    @app.route("/rangs")
    def rangs():
        st = get_state()
        proj = st["runs"][pick_scenario()]
        return render_template("rangs.html", rows=_rows(proj),
                               n=len(proj["teams"]))

    # ------------------------------------------------------------- scenarios
    @app.route("/scenarios")
    def scenarios_page():
        st = get_state()
        cmp_ = sc.compare(st["runs"])
        heads = {k: sc.headline(st["runs"], k) for k in SCENARIOS}
        return render_template("scenarios.html", cmp=cmp_.to_dict("records"),
                               heads=heads, runs=st["runs"],
                               tables={k: _rows(v) for k, v in st["runs"].items()})

    # ------------------------------------------------------------ calendrier
    @app.route("/calendrier")
    def calendrier():
        st = get_state()
        proj = st["runs"][pick_scenario()]
        preds = fixture_predictions(st["model"], proj["fixtures"])
        club = request.args.get("club") or ""
        club = resolve_team(club, proj["teams"]) or ""
        if club:
            preds = preds[(preds.home == club) | (preds.away == club)]
        return render_template("calendrier.html",
                               matchs=preds.to_dict("records"),
                               teams=proj["teams"], club=club)

    # ----------------------------------------------------------------- match
    @app.route("/match")
    def match():
        st = get_state()
        # La prevision d'une affiche ne depend pas de la composition du
        # championnat : on propose les clubs des deux scenarios.
        teams = sc.all_scenario_teams()
        home = resolve_team(request.args.get("dom", ""), teams) or teams[0]
        away = resolve_team(request.args.get("ext", ""), teams)
        if away is None or away == home:
            away = teams[1] if teams[0] == home else teams[0]
        neutral = request.args.get("neutre") == "1"
        pred = st["model"].predict(home, away, neutral=neutral)
        grid, h0, a0 = st["model"].score_grid(home, away, neutral=neutral, span=22)
        best = np.unravel_index(int(np.argmax(grid)), grid.shape)
        return render_template("match.html", teams=teams, home=home, away=away,
                               p=pred, neutral=neutral, grid=grid.tolist(),
                               h0=h0, a0=a0, span=grid.shape[0],
                               best=(h0 + int(best[0]), a0 + int(best[1])),
                               vmax=float(grid.max()))

    # ---------------------------------------------------------------- forces
    @app.route("/forces")
    def forces():
        st = get_state()
        proj = st["runs"][pick_scenario()]
        teams = proj["teams"]
        model = st["model"]
        elo = model.elo.table(teams).set_index("team")
        strength = model.points.strength_table(
            [t for t in teams if model.points.has(t)]).set_index("team")
        rows = []
        for t in teams:
            s = strength.loc[t] if t in strength.index else None
            rows.append({
                "club": t, "elo": float(elo.loc[t, "elo"]),
                "matchs": int(elo.loc[t, "matches"]),
                "attaque": float(s.attaque) if s is not None else None,
                "defense": float(s.defense) if s is not None else None,
                "net": float(s.net) if s is not None else None,
                "conf": model.confidence(t),
            })
        rows.sort(key=lambda r: -r["elo"])
        return render_template("forces.html", rows=rows,
                               base=float(np.exp(model.points.mu)),
                               total=model.league_total,
                               home_pts=model.points.home_advantage_points,
                               home_elo=st["cfg"].elo.home_adv,
                               elo_pt=st["cfg"].elo.elo_per_point,
                               pace=model.points.pace_var,
                               shape=model.points.shape)

    # ------------------------------------------------------------- fiabilite
    @app.route("/fiabilite")
    def fiabilite():
        bt = get_backtest()
        return render_template("fiabilite.html", m=bt["metrics"],
                               rel=bt["reliability"].to_dict("records"),
                               marge=bt["marge"])

    # ------------------------------------------------------------ actualiser
    @app.post("/actualiser")
    def actualiser():
        get_state(force=True)
        return redirect(request.form.get("retour") or url_for("index"))

    return app


def serve(host: str = "127.0.0.1", port: int = 8010, debug: bool = False) -> None:
    create_app().run(host=host, port=port, debug=debug)
