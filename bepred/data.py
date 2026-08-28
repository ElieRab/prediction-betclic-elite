"""Telechargement et normalisation des resultats de Betclic Elite.

Source : les tableaux croises "sports results" de Wikipedia EN, seule source
libre publiant la grille complete des scores (domicile x exterieur) de la
Betclic Elite, mise a jour au fil de la saison.

Limite assumee : la grille donne les scores, pas les dates. L'ordre
chronologique est donc reconstruit par un calendrier canonique en ronde
(methode du cercle) : chaque equipe joue une fois par journee, aller puis
retour. Le *jeu* de matchs joues est exact, seul leur ordre est approche --
ce qui ne deplace l'Elo final qu'a la marge.
"""
from __future__ import annotations

import re
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import (OT_MINUTES, PAGE_PATTERNS, RAW_DIR, REGULATION_MINUTES,
                     WIKI_API, Config)

UA = "bepred/1.0 (projet personnel de prevision Betclic Elite)"
_DASH = "[-‐‑‒–—―−]"

# --------------------------------------------------------------- clubs
#: Regles de normalisation : (motif cherche dans le nom sans accents, nom retenu).
#: L'ordre compte -- "chalons"/"reims" doit passer avant "chalon".
_ALIASES: tuple[tuple[str, str], ...] = (
    ("chalons", "Chalons-Reims"), ("reims", "Chalons-Reims"),
    ("chalon", "Chalon"),
    ("asvel", "ASVEL"), ("villeurbanne", "ASVEL"),
    ("monaco", "Monaco"),
    ("paris", "Paris"),
    ("boulazac", "Boulazac"),
    ("cholet", "Cholet"),
    ("dijon", "Dijon"),
    ("gravelines", "Gravelines-Dunkerque"),
    ("bourg-en-bresse", "JL Bourg"), ("jl bourg", "JL Bourg"),
    ("limoges", "Limoges"),
    ("le mans", "Le Mans"), ("msb", "Le Mans"), ("sarthe", "Le Mans"),
    ("strasbourg", "Strasbourg"), ("sig ", "Strasbourg"),
    ("nancy", "Nancy"), ("sluc", "Nancy"),
    ("nanterre", "Nanterre"),
    ("le portel", "Le Portel"), ("essm", "Le Portel"),
    ("saint-quentin", "Saint-Quentin"), ("st-quentin", "Saint-Quentin"),
    ("pau", "Pau-Lacq-Orthez"), ("orthez", "Pau-Lacq-Orthez"),
    ("roanne", "Roanne"),
    ("blois", "Blois"),
    ("rochel", "La Rochelle"),
    ("metropolitans", "Metropolitans 92"), ("levallois", "Metropolitans 92"),
    ("boulogne", "Metropolitans 92"),
    ("fos", "Fos Provence"),
    ("antibes", "Antibes"),
    ("orleans", "Orleans"),
    ("vichy", "Vichy-Clermont"),
    ("nantes", "Nantes"),
    ("le havre", "Le Havre"),
    ("evreux", "Evreux"),
    ("poitiers", "Poitiers"),
    ("denain", "Denain"),
    ("aix", "Aix-Maurienne"), ("maurienne", "Aix-Maurienne"),
    ("quimper", "Quimper"),
    ("caen", "Caen"),
    ("hyeres", "Hyeres-Toulon"), ("toulon", "Hyeres-Toulon"),
    ("rouen", "Rouen"),
    ("chartres", "Chartres"),
    ("saint-vallier", "Saint-Vallier"),
    ("lille", "Lille"),
)

#: Couleur d'accent par club, pour les pastilles de l'interface.
_COLORS = {
    "ASVEL": "#c8102e", "Monaco": "#d3172d", "Paris": "#2b6cb0",
    "Le Mans": "#d21b25", "Cholet": "#e2001a", "Dijon": "#f0b323",
    "Nanterre": "#0b6b3a", "JL Bourg": "#0b3d91", "Limoges": "#00713c",
    "Strasbourg": "#0057b8", "Chalon": "#e2001a", "Nancy": "#c8102e",
    "Gravelines-Dunkerque": "#f47b20", "Boulazac": "#1a3b73",
    "Le Portel": "#0b6ab0", "Saint-Quentin": "#e4002b",
    "Pau-Lacq-Orthez": "#00843d", "Roanne": "#7ac143",
    "Blois": "#e2001a", "La Rochelle": "#f9c000",
    "Metropolitans 92": "#0b3d91", "Chalons-Reims": "#e2001a",
}

#: Betclic Elite 2026-2027 : 15 clubs certains, le 16e depend du sort de Monaco.
#: La DNCCG puis la chambre d'appel ont refuse l'engagement de l'AS Monaco
#: (1er aout 2026) ; si le refus est confirme, Saint-Quentin -- relegue
#: sportivement -- serait repeche.
SEASON_2026_CORE = (
    "ASVEL", "Boulazac", "Chalon", "Cholet", "Dijon", "Gravelines-Dunkerque",
    "JL Bourg", "Le Mans", "Limoges", "Nancy", "Nanterre", "Paris",
    "Pau-Lacq-Orthez", "Roanne", "Strasbourg",
)
SCENARIOS: dict[str, dict] = {
    "monaco": {
        "libelle": "Avec Monaco, sans Saint-Quentin",
        "court": "Monaco",
        "seizieme": "Monaco",
        "resume": "L'AS Monaco obtient gain de cause devant le CNOSF et conserve "
                  "sa place ; Saint-Quentin reste en Élite 2.",
    },
    "saint-quentin": {
        "libelle": "Avec Saint-Quentin, sans Monaco",
        "court": "Saint-Quentin",
        "seizieme": "Saint-Quentin",
        "resume": "Le refus d'engagement de Monaco est confirmé ; Saint-Quentin, "
                  "relégué sportivement, est repêché en Betclic Élite.",
    },
}


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_team(raw: str) -> str:
    """Ramene les variantes d'un club ('LDLC ASVEL', 'ASVEL Basket') a un nom."""
    key = _fold(raw)
    for needle, canon in _ALIASES:
        if needle in key:
            return canon
    return str(raw).strip()


def slug(name: str) -> str:
    """Nom de club utilisable dans une URL ('Pau-Lacq-Orthez' -> 'pau-lacq-orthez')."""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", _fold(name))).strip("-")


def display_name(team: str) -> str:
    return team


def team_color(team: str) -> str:
    return _COLORS.get(team, "#7b8bb0")


# ------------------------------------------------------- calendrier canonique
def round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Calendrier aller-retour par la methode du cercle.

    Renvoie une liste de journees, chacune etant une liste de (domicile, ext).
    L'alternance domicile/exterieur evite les series de matchs a sens unique.
    """
    ts = list(teams)
    if len(ts) % 2:
        ts.append(None)
    n = len(ts)
    half = n // 2
    order = ts[:]
    aller: list[list[tuple[str, str]]] = []
    for r in range(n - 1):
        day = []
        for i in range(half):
            a, b = order[i], order[n - 1 - i]
            if a is None or b is None:
                continue
            day.append((a, b) if (r + i) % 2 == 0 else (b, a))
        aller.append(day)
        order = [order[0], order[-1]] + order[1:-1]
    retour = [[(b, a) for a, b in day] for day in aller]
    return aller + retour


def season_calendar(teams: list[str], season: int) -> dict[tuple[str, str], date]:
    """Pseudo-date de chaque affiche : les journees etalees de septembre a mai."""
    days = round_robin(sorted(teams))
    start, end = date(season, 9, 20), date(season + 1, 5, 10)
    span = (end - start).days
    out: dict[tuple[str, str], date] = {}
    for i, day in enumerate(days):
        when = start + timedelta(days=round(span * i / max(1, len(days) - 1)))
        for home, away in day:
            out[(home, away)] = when
    return out


# ------------------------------------------------------------- telechargement
def _wikitext(title: str, session: requests.Session) -> str:
    r = session.get(WIKI_API, params={
        "action": "parse", "page": title, "prop": "wikitext",
        "format": "json", "formatversion": "2"}, timeout=60)
    r.raise_for_status()
    return r.json().get("parse", {}).get("wikitext", "")


def fetch_season_wikitext(season: int, session: requests.Session) -> str:
    """Recupere la page de la saison, quel que soit son intitule (Elite / Pro A)."""
    a, b = season, f"{(season + 1) % 100:02d}"
    for pattern in PAGE_PATTERNS:
        try:
            text = _wikitext(pattern.format(a=a, b=b), session)
        except requests.RequestException:
            text = ""
        if len(text) > 2000:
            return text
        time.sleep(0.5)
    return ""


def parse_results(text: str, season: int) -> pd.DataFrame:
    """Extrait la grille des scores d'une page de saison."""
    codes: dict[str, str] = {}
    for code, raw in re.findall(r"\|\s*name_([A-Za-z0-9]{2,4})\s*=\s*(.+)", text):
        value = raw.split("<")[0].strip().strip("|").strip()
        link = re.match(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", value)
        if link:
            value = (link.group(2) or link.group(1)).strip()
        if value:
            codes[code] = normalize_team(value)

    overtime = {(m.group(1), m.group(2)) for m in re.finditer(
        r"result_([A-Za-z0-9]{2,4})_([A-Za-z0-9]{2,4})\s*=\s*OT", text)}

    rows = []
    pattern = (r"\|\s*match_([A-Za-z0-9]{2,4})_([A-Za-z0-9]{2,4})\s*=\s*"
               r"(\d{1,3})\s*" + _DASH + r"\s*(\d{1,3})")
    for m in re.finditer(pattern, text):
        ch, ca = m.group(1), m.group(2)
        if ch not in codes or ca not in codes or codes[ch] == codes[ca]:
            continue
        rows.append({
            "season": season,
            "home": codes[ch], "away": codes[ca],
            "hp": int(m.group(3)), "ap": int(m.group(4)),
            "ot": int((ch, ca) in overtime),
        })
    df = pd.DataFrame(rows, columns=["season", "home", "away", "hp", "ap", "ot"])
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["home", "away"], keep="first")
    teams = sorted(set(df.home) | set(df.away) | set(codes.values()))
    cal = season_calendar(teams, season)
    df["date"] = [cal.get((h, a), date(season + 1, 1, 1))
                  for h, a in zip(df.home, df.away)]
    return df.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def _raw_path(season: int) -> Path:
    return RAW_DIR / f"elite_{season}.csv"


def refresh(cfg: Config | None = None, seasons: list[int] | None = None) -> dict:
    """Retelecharge les saisons demandees et reecrit le cache CSV.

    Renvoie {saison: nb de matchs}. -1 signale une page indisponible dont le
    cache local a ete conserve.
    """
    cfg = cfg or Config.load()
    seasons = seasons or cfg.seasons_to_load()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[int, int] = {}
    with requests.Session() as s:
        s.headers.update({"User-Agent": UA})
        for season in seasons:
            text = fetch_season_wikitext(season, s)
            df = parse_results(text, season) if text else pd.DataFrame()
            if df.empty:
                # Saison sans grille publiee (a venir, ou page absente) : on
                # garde le cache existant plutot que de l'effacer.
                report[season] = -1 if _raw_path(season).exists() else 0
                continue
            df.to_csv(_raw_path(season), index=False, encoding="utf-8")
            report[season] = len(df)
            time.sleep(0.5)
    return report


EMPTY = ["season", "date", "home", "away", "hp", "ap", "ot"]


def load_matches(cfg: Config | None = None, force: bool = False) -> pd.DataFrame:
    """Charge l'historique complet, en telechargeant ce qui manque."""
    cfg = cfg or Config.load()
    seasons = cfg.seasons_to_load()
    if force or not all(_raw_path(s).exists() for s in seasons[:-1]):
        refresh(cfg, seasons)
    frames = []
    for season in seasons:
        path = _raw_path(season)
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8")
        df["date"] = pd.to_datetime(df["date"]).dt.date
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=EMPTY)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------- helpers
def regulation_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Ramene les scores des matchs en prolongation a 40 minutes.

    Une prolongation ajoute 5 minutes de jeu : sans correction, ces matchs
    tirent vers le haut le niveau de marque estime par le modele Poisson.
    """
    out = df.copy()
    factor = REGULATION_MINUTES / (REGULATION_MINUTES + OT_MINUTES)
    scale = np.where(out.ot.to_numpy(int) == 1, factor, 1.0)
    out["hp_reg"] = out.hp.to_numpy(float) * scale
    out["ap_reg"] = out.ap.to_numpy(float) * scale
    return out


def season_teams(matches: pd.DataFrame, season: int) -> list[str]:
    df = matches[matches.season == season]
    return sorted(set(df.home) | set(df.away))


def resolve_team(name: str, teams: list[str]) -> str | None:
    """Retrouve un club a partir d'une saisie libre."""
    if not name:
        return None
    key = _fold(name)
    for t in teams:
        if _fold(t) == key:
            return t
    guess = normalize_team(name)
    if guess in teams:
        return guess
    hits = [t for t in teams if key in _fold(t)]
    return hits[0] if len(hits) == 1 else None


def scenario_teams(key: str) -> list[str]:
    """Les 16 clubs d'un scenario de composition 2026-2027."""
    return sorted([*SEASON_2026_CORE, SCENARIOS[key]["seizieme"]])
