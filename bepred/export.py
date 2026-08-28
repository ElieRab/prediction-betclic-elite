"""Export du site en pages statiques.

L'application n'a aucune saisie libre : deux scenarios, seize clubs, une liste
finie d'affiches. Tout son espace d'etats tient donc dans quelques centaines
de fichiers, qu'un hebergement statique sert instantanement et gratuitement.

Les pages sont produites par l'application elle-meme, via son client de test :
memes vues, memes gabarits. Seule la fabrication des liens change -- `url_for`
est remplace par une version qui pointe vers des fichiers, en chemins
relatifs, pour que le site fonctionne aussi bien a la racine d'un domaine que
dans un sous-repertoire.
"""
from __future__ import annotations

import posixpath
import shutil
from pathlib import Path

from flask import request

from . import scenarios as scn
from .data import SCENARIOS, slug
from .web import DEFAULT_SCENARIO, create_app, get_state

#: Page d'accueil du site : le scenario affiche par defaut.
HOME = DEFAULT_SCENARIO


def page_path(endpoint: str, values) -> str:
    """Fichier produit pour une vue et ses parametres."""
    get = values.get
    key = get("sc") or HOME
    if key not in SCENARIOS:
        key = HOME
    if endpoint == "match":
        # La prevision d'une affiche ne depend pas du seizieme club : une
        # seule serie de pages sert les deux scenarios.
        dom, ext = get("dom"), get("ext")
        if not dom or not ext:
            return "match/index.html"
        suffix = "-neutre" if str(get("neutre")) == "1" else ""
        return f"match/{slug(dom)}--{slug(ext)}{suffix}.html"
    if endpoint == "calendrier":
        club = get("club")
        return f"{key}/calendrier/{slug(club)}.html" if club else f"{key}/calendrier.html"
    pages = {
        "index": "index.html",
        "rangs": "rangs.html",
        "scenarios_page": "scenarios.html",
        "forces": "forces.html",
        "fiabilite": "fiabilite.html",
    }
    if endpoint not in pages:
        # Mieux vaut interrompre l'export qu'exporter un lien qui retombe
        # silencieusement sur l'accueil.
        raise KeyError(f"aucun fichier prevu pour la vue « {endpoint} »")
    return f"{key}/{pages[endpoint]}"


def _static_url_for(endpoint: str, **values) -> str:
    """Remplacant de `url_for` : un chemin de fichier, relatif a la page."""
    if endpoint == "static":
        target = "static/" + values.get("filename", "")
    elif endpoint == "actualiser":
        return "#"
    else:
        target = page_path(endpoint, values)
    here = posixpath.dirname(page_path(request.endpoint, request.args))
    return posixpath.relpath(target, here or ".")


REDIRECTION = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={cible}">
<link rel="canonical" href="{cible}">
<title>Betclic Élite – prédiction</title></head>
<body><p>Redirection vers <a href="{cible}">la projection</a>…</p></body></html>
"""


def build(out: Path, n_sims: int = 10000, actualiser: bool = True,
          verbose: bool = True) -> dict:
    """Genere l'integralite du site dans `out`. Renvoie un petit rapport."""
    out = Path(out)
    app = create_app()
    app.jinja_env.globals["url_for"] = _static_url_for
    app.jinja_env.globals["statique"] = True

    state = get_state(force=actualiser, n_sims=n_sims)
    if state["matches"].empty:
        raise RuntimeError("aucun resultat charge : export interrompu")

    if out.exists():
        shutil.rmtree(out)
    (out / "static").mkdir(parents=True)
    shutil.copy(Path(__file__).parent / "static" / "style.css", out / "static")

    client = app.test_client()
    written = 0

    def render(url: str, *destinations: str) -> None:
        nonlocal written
        response = client.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"{url} a repondu {response.status_code}")
        html = response.get_data(as_text=True)
        for dest in destinations:
            path = out / dest
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(html, encoding="utf-8")
            written += 1

    for key in SCENARIOS:
        teams = state["runs"][key]["teams"]
        render(f"/?sc={key}", f"{key}/index.html")
        render(f"/rangs?sc={key}", f"{key}/rangs.html")
        render(f"/scenarios?sc={key}", f"{key}/scenarios.html")
        render(f"/fiabilite?sc={key}", f"{key}/fiabilite.html")
        render(f"/forces?sc={key}", f"{key}/forces.html")
        render(f"/calendrier?sc={key}", f"{key}/calendrier.html")
        for club in teams:
            render(f"/calendrier?sc={key}&club={club}",
                   f"{key}/calendrier/{slug(club)}.html")
        if verbose:
            print(f"  {key} : {len(teams) + 6} pages")

    union = scn.all_scenario_teams()
    default = _default_pair(state, union)
    for home in union:
        for away in union:
            if home == away:
                continue
            base = f"match/{slug(home)}--{slug(away)}"
            extra = ["match/index.html"] if (home, away) == default else []
            render(f"/match?dom={home}&ext={away}", base + ".html", *extra)
            render(f"/match?dom={home}&ext={away}&neutre=1", base + "-neutre.html")
    if verbose:
        print(f"  affiches : {len(union) * (len(union) - 1) * 2} pages")

    (out / "index.html").write_text(
        REDIRECTION.format(cible=f"{HOME}/index.html"), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    written += 2

    taille = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    return {"fichiers": written, "octets": taille,
            "saison": state["season"], "matchs": len(state["matches"])}


def _default_pair(state: dict, union: list[str]) -> tuple[str, str]:
    """Affiche mise en avant : les deux meilleurs Elo du moment."""
    table = state["model"].elo.table(union)
    if len(table) < 2:
        return union[0], union[1]
    return str(table.team.iloc[0]), str(table.team.iloc[1])


def main(out: str = "site", n_sims: int = 10000, actualiser: bool = True) -> None:
    report = build(Path(out), n_sims=n_sims, actualiser=actualiser)
    print(f"\n{report['fichiers']} fichiers, "
          f"{report['octets'] / 1e6:.1f} Mo, dans « {out} »")
