"""
Utilities for working with StatsBomb competitions.

Parses the local competitions.json and exposes convenient mappings so the
pipeline can flexibly target different leagues and seasons.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import importlib.util


# Absolute path anchor: workspace root passed from user context.
WORKSPACE_ROOT = Path("/Users/michae14/Soccer Analytics/Soccer Analytics Pipeline")
COMPETITIONS_JSON = WORKSPACE_ROOT / "competitions.json"


def _load_competitions() -> List[Dict]:
    if not COMPETITIONS_JSON.exists():
        raise FileNotFoundError(f"competitions.json not found at {COMPETITIONS_JSON}")
    with COMPETITIONS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_static_competition_mapping() -> Dict[str, int]:
    """Load COMPETITION_ID_MAPPING from Pipeline/utils/competitions_id.py robustly."""
    # 1) Try package-style import
    try:
        from .competitions_id import COMPETITION_ID_MAPPING as mapping  # type: ignore
        return dict(mapping)
    except Exception:
        pass

    try:
        from Pipeline.utils.competitions_id import COMPETITION_ID_MAPPING as mapping  # type: ignore
        return dict(mapping)
    except Exception:
        pass

    # 2) Fallback: import by file path relative to this file
    utils_path = Path(__file__).parent / "competitions_id.py"
    if utils_path.exists():
        spec = importlib.util.spec_from_file_location("competitions_id", str(utils_path))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
        mapping = getattr(module, "COMPETITION_ID_MAPPING", {})
        if isinstance(mapping, dict):
            return dict(mapping)

    raise ImportError("Unable to load static COMPETITION_ID_MAPPING from Pipeline/utils/competitions_id.py")


def get_competition_id_mapping() -> Dict[str, int]:
    """
    Returns mapping: competition_name -> competition_id from the static mapping file
    Pipeline/utils/competitions_id.py.
    """
    return _load_static_competition_mapping()


def get_competition_key_mapping() -> Dict[str, Tuple[int, str]]:
    """
    Returns mapping: "{competition_name}" (string key) -> (competition_id, country_name)
    Useful for displaying additional context.
    """
    data = _load_competitions()
    mapping: Dict[str, Tuple[int, str]] = {}
    for row in data:
        name = row.get("competition_name")
        cid = row.get("competition_id")
        country = row.get("country_name", "")
        if name and cid is not None:
            mapping[name] = (int(cid), country)
    return mapping


def get_seasons_for_competition(competition: Union[str, int]) -> Dict[str, int]:
    """
    Returns mapping for a given competition: season_name -> season_id.
    competition can be competition_name or competition_id.
    """
    data = _load_competitions()
    if isinstance(competition, str):
        # map name to id first
        name_to_id = get_competition_id_mapping()
        comp_id = name_to_id.get(competition)
        if comp_id is None:
            return {}
    else:
        comp_id = int(competition)

    seasons: Dict[str, int] = {}
    for row in data:
        if int(row.get("competition_id", -1)) == comp_id:
            season_name = row.get("season_name")
            season_id = row.get("season_id")
            if season_name and season_id is not None:
                seasons[season_name] = int(season_id)
    return seasons


def resolve_competition_id(competition: Optional[Union[str, int]]) -> Optional[int]:
    if competition is None:
        return None
    if isinstance(competition, int):
        return competition
    mapping = get_competition_id_mapping()
    return mapping.get(competition)


def resolve_season_id(competition: Optional[Union[str, int]], season: Optional[Union[str, int]]) -> Optional[int]:
    if season is None:
        return None
    if isinstance(season, int):
        return season
    seasons = get_seasons_for_competition(competition if competition is not None else "")
    return seasons.get(season)


# Eager constant for simple imports
try:
    COMPETITION_ID_MAPPING: Dict[str, int] = get_competition_id_mapping()
except Exception:
    # Allow imports to succeed even if file missing during certain execution contexts
    COMPETITION_ID_MAPPING = {}