"""Ligne de commande."""
from __future__ import annotations

import argparse

import pandas as pd

from . import scenarios as sc
from .backtest import margin_check, metrics, reliability, walk_forward
from .config import Config, current_season, season_label
from .data import SCENARIOS, load_matches, refresh
from .model import BEModel
from .season import current_table
from .tune import search


def _projection(args) -> None:
    cfg = Config.load()
    matches = load_matches(cfg, force=args.actualiser)
    model = BEModel.build(matches, cfg)
    season = current_season()
    runs = sc.run(model, matches, cfg, season, n_sims=args.sims)
    for key in ([args.scenario] if args.scenario else list(SCENARIOS)):
        proj = runs[key]
        print()
        print(f"=== {SCENARIOS[key]['libelle']} - {season_label(season)} ===")
        df = pd.DataFrame({
            "club": proj["teams"],
            "V": proj["wins_mean"].round(1),
            "diff": proj["diff_mean"].round(0),
            "top6 %": (100 * proj["p_top"]).round(1),
            "playoffs %": (100 * proj["p_playoffs"]).round(1),
            "titre %": (100 * proj["p_title"]).round(1),
            "releg. %": (100 * proj["p_rel"]).round(1),
        }).sort_values("V", ascending=False)
        print(df.to_string(index=False))


def _compare(args) -> None:
    cfg = Config.load()
    matches = load_matches(cfg)
    model = BEModel.build(matches, cfg)
    runs = sc.run(model, matches, cfg, current_season(), n_sims=args.sims)
    out = sc.compare(runs)[["club", "wins_mean_a", "wins_mean_b", "wins_mean_d",
                            "p_top_d", "p_playoffs_d", "p_rel_d"]].copy()
    for c in ("p_top_d", "p_playoffs_d", "p_rel_d"):
        out[c] = (100 * out[c]).round(1)
    for c in ("wins_mean_a", "wins_mean_b", "wins_mean_d"):
        out[c] = out[c].round(2)
    out.columns = ["club", "V Monaco", "V St-Q.", "d V", "d top6", "d PO", "d releg."]
    print(out.to_string(index=False))


def _match(args) -> None:
    cfg = Config.load()
    matches = load_matches(cfg)
    model = BEModel.build(matches, cfg)
    model.prepare_season(current_season(), sc.all_scenario_teams())
    p = model.predict(args.domicile, args.exterieur)
    print(f"{args.domicile} - {args.exterieur}")
    print(f"  score attendu : {p['lam_h']:.1f} - {p['lam_a']:.1f}")
    print(f"  victoire dom. : {100 * p['p_home']:.1f} %"
          f"   ext. : {100 * p['p_away']:.1f} %")
    print(f"  ecart {p['marge']:+.1f} | total {p['total']:.0f}"
          f" | prolongation {100 * p['p_ot']:.1f} %")


def _fiabilite(args) -> None:
    cfg = Config.load()
    pred = walk_forward(load_matches(cfg), cfg)
    for k, v in metrics(pred).items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    print()
    print(reliability(pred).to_string(index=False))
    print()
    print(margin_check(pred))


def _regler(args) -> None:
    cfg = Config.load()
    best, score = search(load_matches(cfg), cfg, rounds=args.rounds)
    print()
    print("Meilleure configuration :",
          {k: round(v, 4) for k, v in score.items() if isinstance(v, float)})
    best.save()
    print("Enregistree dans data/params.json")


def _table(args) -> None:
    cfg = Config.load()
    matches = load_matches(cfg, force=args.actualiser)
    season = args.saison or current_season()
    print(current_table(matches, season).to_string(index=False))


def _exporter(args) -> None:
    from .export import build
    report = build(args.sortie, n_sims=args.sims, actualiser=not args.hors_ligne)
    print()
    print(f"{report['fichiers']} fichiers, {report['octets'] / 1e6:.1f} Mo,"
          f" dans « {args.sortie} »")


def _web(args) -> None:
    from .web import serve
    serve(host=args.host, port=args.port, debug=args.debug)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="bepred", description="Prevision de la Betclic Elite (Elo + Poisson)")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("projection", help="classement projete")
    q.add_argument("--sims", type=int, default=10000)
    q.add_argument("--scenario", choices=list(SCENARIOS))
    q.add_argument("--actualiser", action="store_true")
    q.set_defaults(func=_projection)

    q = sub.add_parser("scenarios", help="comparer Monaco et Saint-Quentin")
    q.add_argument("--sims", type=int, default=10000)
    q.set_defaults(func=_compare)

    q = sub.add_parser("match", help="prevision d'une affiche")
    q.add_argument("domicile")
    q.add_argument("exterieur")
    q.set_defaults(func=_match)

    q = sub.add_parser("fiabilite", help="validation glissante")
    q.set_defaults(func=_fiabilite)

    q = sub.add_parser("regler", help="regler les hyperparametres")
    q.add_argument("--rounds", type=int, default=2)
    q.set_defaults(func=_regler)

    q = sub.add_parser("classement", help="classement reel a date")
    q.add_argument("--saison", type=int)
    q.add_argument("--actualiser", action="store_true")
    q.set_defaults(func=_table)

    q = sub.add_parser("actualiser", help="retelecharger les resultats")
    q.set_defaults(func=lambda a: print(refresh()))

    q = sub.add_parser("exporter", help="generer le site statique")
    q.add_argument("--sortie", default="site")
    q.add_argument("--sims", type=int, default=10000)
    q.add_argument("--hors-ligne", action="store_true",
                   help="utiliser le cache local au lieu de retelecharger")
    q.set_defaults(func=_exporter)

    q = sub.add_parser("web", help="interface web")
    q.add_argument("--host", default="127.0.0.1")
    q.add_argument("--port", type=int, default=8010)
    q.add_argument("--debug", action="store_true")
    q.set_defaults(func=_web)

    args = p.parse_args(argv)
    args.func(args)
