"""Source officielle : l'API publique du site de la LNB.

Wikipedia publie la grille complete des saisons passees, mais elle n'est
alimentee qu'au bon vouloir des contributeurs : au 3 octobre 2026, la page
2026-27 existait avec ses 240 cellules... toutes vides, alors que deux
journees avaient ete jouees. La saison en cours vient donc de l'API que le
site lnb.fr interroge lui-meme.

Deux avantages par rapport a Wikipedia : les resultats arrivent sans delai, et
chaque match porte sa **date reelle** et son numero de journee -- de quoi
rejouer l'Elo dans le bon ordre au lieu du calendrier reconstruit.

Une limite : l'API ne renvoie qu'une fenetre glissante d'environ cinq semaines
autour du jour courant, et ignore les parametres de date. Les resultats sont
donc **accumules** dans le cache CSV a chaque passage ; un rafraichissement
quotidien ne peut rien manquer.

Rien ici n'est authentifie au sens propre : le jeton est delivre a tout
visiteur par lnb.fr/api/token, et les donnees lues sont celles qu'affiche la
page publique du calendrier. C'est en revanche une interface non documentee,
qui peut changer sans preavis -- d'ou le repli sur Wikipedia.
"""
from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import requests

TOKEN_URL = "https://lnb.fr/api/token"
API = "https://api-prod.lnb.fr"

#: Division 1 = Betclic Elite + Supercoupe ; l'abreviation separe les deux.
DIVISION_ELITE = 1
ABBREV_CHAMPIONNAT = "PROA"      # SUP = Supercoupe, PROB/ELI2 = deuxieme division

HEADERS = {
    "User-Agent": "bepred/1.1 (projet personnel de prevision Betclic Elite)",
    "Origin": "https://lnb.fr",
    "Referer": "https://lnb.fr/fr/calendar",
}


class LNBError(RuntimeError):
    """L'API officielle n'a pas repondu comme attendu."""


def _token(session: requests.Session) -> str:
    r = session.get(TOKEN_URL, timeout=30)
    r.raise_for_status()
    token = r.json().get("token")
    if not token:
        raise LNBError("jeton absent de la reponse de lnb.fr/api/token")
    return token


def _post(session: requests.Session, token: str, route: str, body: dict) -> dict:
    r = session.post(f"{API}/{route}", data=json.dumps(body), timeout=40,
                     headers={"Authorization": f"Bearer {token}",
                              "Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()


def _get(session: requests.Session, token: str, route: str, params: dict | None = None) -> dict:
    r = session.get(f"{API}/{route}", params=params or {}, timeout=40,
                    headers={"Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------------ lectures
def fetch_matches(season: int, session: requests.Session | None = None) -> list[dict]:
    """Matchs de championnat presents dans la fenetre renvoyee par l'API."""
    own = session is None
    session = session or requests.Session()
    if own:
        session.headers.update(HEADERS)
    try:
        token = _token(session)
        payload = _post(session, token, "match/v3/getCalendar",
                        {"division_external_id": DIVISION_ELITE, "year": season})
        groups = payload.get("data") or []
        return [m for g in groups for m in (g.get("data") or [])
                if m.get("competition_abbrev") == ABBREV_CHAMPIONNAT]
    finally:
        if own:
            session.close()


def fetch_teams(season: int, session: requests.Session | None = None) -> list[str]:
    """Liste officielle des clubs engages, normalisee."""
    from .data import normalize_team
    own = session is None
    session = session or requests.Session()
    if own:
        session.headers.update(HEADERS)
    try:
        token = _token(session)
        main = _get(session, token, "competition/getMainCompetition", {"year": season})
        comps = [c for c in (main.get("data") or [])
                 if c.get("competition_abbrev") == ABBREV_CHAMPIONNAT]
        if not comps:
            return []
        teams = _get(session, token, "competition/getCompetitionTeams",
                     {"competition_external_id": comps[0]["external_id"]})
        return sorted({normalize_team(t["team_name"])
                       for t in (teams.get("data") or []) if t.get("team_name")})
    finally:
        if own:
            session.close()


# ------------------------------------------------------------------ parsing
COLUMNS = ["season", "date", "home", "away", "hp", "ap", "ot", "journee"]


def parse_matches(raw: list[dict], season: int) -> pd.DataFrame:
    """Ne garde que les rencontres terminees, avec leur score et leur date.

    `ot` vaut toujours 0 : l'API ne distingue pas les prolongations sur cette
    route. L'effet est mineur (environ 4 % des matchs, corriges de cinq
    minutes de jeu), mais il est reel et assume.
    """
    rows = []
    for m in raw:
        teams = m.get("teams") or []
        if m.get("match_status") != "COMPLETE" or len(teams) != 2:
            continue
        try:
            hp = int(str(teams[0].get("score_string", "")).strip())
            ap = int(str(teams[1].get("score_string", "")).strip())
        except (TypeError, ValueError):
            continue
        from .data import normalize_team
        rows.append({
            "season": season,
            "date": _as_date(m.get("match_date")),
            "home": normalize_team(teams[0].get("team_name", "")),
            "away": normalize_team(teams[1].get("team_name", "")),
            "hp": hp, "ap": ap, "ot": 0,
            "journee": int(m.get("round_number") or 0),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)
    if df.empty:
        return df
    df = df[df.home != df.away]
    df = df.drop_duplicates(subset=["home", "away"], keep="last")
    return df.sort_values(["date", "home"], kind="stable").reset_index(drop=True)


def fixtures(raw: list[dict], season: int) -> pd.DataFrame:
    """Rencontres a venir connues de l'API (fenetre glissante seulement)."""
    from .data import normalize_team
    rows = []
    for m in raw:
        teams = m.get("teams") or []
        if m.get("match_status") == "COMPLETE" or len(teams) != 2:
            continue
        rows.append({
            "season": season, "date": _as_date(m.get("match_date")),
            "home": normalize_team(teams[0].get("team_name", "")),
            "away": normalize_team(teams[1].get("team_name", "")),
            "journee": int(m.get("round_number") or 0),
        })
    df = pd.DataFrame(rows, columns=["season", "date", "home", "away", "journee"])
    return df[df.home != df.away].reset_index(drop=True) if not df.empty else df


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return date.today()


def rounds(season: int, session: requests.Session | None = None) -> dict[int, date]:
    """Date de reference de chaque journee du championnat."""
    own = session is None
    session = session or requests.Session()
    if own:
        session.headers.update(HEADERS)
    try:
        token = _token(session)
        main = _get(session, token, "competition/getMainCompetition", {"year": season})
        comps = [c for c in (main.get("data") or [])
                 if c.get("competition_abbrev") == ABBREV_CHAMPIONNAT]
        if not comps:
            return {}
        data = _get(session, token, "competition/getCompetitionRounds",
                    {"competition_external_id": comps[0]["external_id"],
                     "exclude_date_filter": "yes"})
        return {int(r["round_number"]): _as_date(r.get("round_date"))
                for r in (data.get("data") or []) if r.get("round_number")}
    finally:
        if own:
            session.close()
