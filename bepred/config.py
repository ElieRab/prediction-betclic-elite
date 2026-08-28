"""Configuration et hyperparametres du modele Betclic Elite."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARAMS_FILE = DATA_DIR / "params.json"

#: Les scores match par match viennent des tableaux croises de Wikipedia EN
#: (modele "sports results"), seule source libre qui publie la grille complete
#: de la Betclic Elite. Le titre de la page change selon l'annee.
WIKI_API = "https://en.wikipedia.org/w/api.php"
PAGE_PATTERNS = (
    "{a}\u2013{b} LNB \u00c9lite season",
    "{a}\u2013{b} Pro A season",
)

#: Saisons chargees par defaut. Les pages anterieures a 2023-24 n'ont pas de
#: grille de resultats exploitable (2020-21 a 2022-23) : l'historique utile
#: commence donc a 2023-24.
HISTORY_SEASONS = (2023, 2024, 2025)

TEAMS_PER_SEASON = 16       # Betclic Elite : 16 clubs, 30 journees
REGULATION_MINUTES = 40.0
OT_MINUTES = 5.0


def season_label(start_year: int) -> str:
    return f"{start_year}-{start_year + 1}"


def current_season(today: date | None = None) -> int:
    """Annee de debut de la saison en cours (bascule au 1er juillet)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


@dataclass
class EloParams:
    k: float = 24.0                 # vitesse d'ajustement
    home_adv: float = 100.0         # avantage du terrain, en points Elo
    season_regress: float = 0.25    # regression vers la moyenne a l'intersaison
    start_rating: float = 1500.0
    new_team_penalty: float = 90.0  # decote d'un club jamais vu
    promotion_shift: float = 60.0   # ajustement au changement de division
    mov_a: float = 0.80             # exposant du multiplicateur de marge
    mov_b: float = 7.5              # denominateur du multiplicateur de marge
    elo_per_point: float = 28.0     # points Elo valant 1 point d'ecart au score


@dataclass
class PoissonParams:
    half_life_days: float = 240.0   # demi-vie de la ponderation temporelle
    ridge: float = 6.0              # regularisation L2 sur attaque/defense
    max_points: int = 140           # taille de la grille de scores
    pace_var: float = 0.0           # variance du facteur de rythme partage
    fit_pace: bool = True           # estimer pace_var sur les donnees
    level_window: int = 120         # matchs servant a caler le niveau de marque


@dataclass
class BlendParams:
    poisson_weight: float = 0.70    # part du Poisson dans l'ecart attendu
    shrink_matches: float = 20.0    # nb de matchs pour croire l'attaque/defense
    ot_slope: float = 0.35          # attenuation de l'avantage en prolongation


@dataclass
class Zones:
    """Zones de classement de la Betclic Elite (16 clubs)."""
    direct: int = 6        # 1-6  : playoffs directs
    playin: int = 10       # 7-10 : play-in
    relegation: int = 15   # 15-16 : relegation en Elite 2


@dataclass
class Config:
    seasons: tuple[int, ...] = HISTORY_SEASONS
    elo: EloParams = field(default_factory=EloParams)
    poisson: PoissonParams = field(default_factory=PoissonParams)
    blend: BlendParams = field(default_factory=BlendParams)
    zones: Zones = field(default_factory=Zones)

    def seasons_to_load(self, today: date | None = None) -> list[int]:
        """Historique + saison en cours (meme si sa page n'existe pas encore)."""
        cur = current_season(today)
        return sorted({*self.seasons, cur})

    # ---------------------------------------------------------------- io
    def save(self, path: Path = PARAMS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = PARAMS_FILE) -> "Config":
        cfg = cls()
        if not path.exists():
            return cfg
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "seasons" in raw:
            cfg.seasons = tuple(int(s) for s in raw["seasons"])
        for key, klass in (("elo", EloParams), ("poisson", PoissonParams),
                           ("blend", BlendParams), ("zones", Zones)):
            if key in raw:
                setattr(cfg, key, klass(**{k: v for k, v in raw[key].items()
                                           if k in klass.__dataclass_fields__}))
        return cfg
