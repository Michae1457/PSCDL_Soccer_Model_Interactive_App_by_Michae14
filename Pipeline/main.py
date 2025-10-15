"""
Main Soccer Analytics Pipeline Orchestration Script

Interactive orchestrator for:
  1) 01_data_preparation.py (scrape)
  2) 02_player_level_transformer.py (transform)
  3) 03_statistical_aggregator.py (aggregate)

Author: Michael Xu
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import importlib.util

# Resolve paths
current_dir = Path(__file__).parent

# Load numbered modules from sibling files
def load_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, str(current_dir / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module

data_prep = load_module("data_preparation", "01_data_preparation.py")
player_transformer = load_module("player_transformer", "02_player_level_transformer.py")
statistical_agg = load_module("statistical_aggregator", "03_statistical_aggregator.py")

# Load competitions utils
try:
    import Pipeline.utils.competitions_utils as comp_utils
except ImportError:
    # Fallback to relative path
    utils_path = current_dir / "utils" / "competitions_utils.py"
    spec = importlib.util.spec_from_file_location("competitions_utils", str(utils_path))
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    comp_utils = module  # type: ignore


def prompt_competition_and_season() -> tuple[str, int, List[str]]:
    static_mapping = getattr(comp_utils, "COMPETITION_ID_MAPPING", {})
    names = list(static_mapping.keys())
    if not names:
        raise RuntimeError("No competitions found in COMPETITION_ID_MAPPING")

    print("\n📋 Available Competitions:")
    print("-" * 40)
    for idx, name in enumerate(names, 1):
        print(f"{idx:2d}. {name}")

    while True:
        ans = input(f"\n🎯 Select competition (1-{len(names)}): ").strip()
        try:
            choice = int(ans)
            if 1 <= choice <= len(names):
                break
        except Exception:
            pass
        print("Please enter a valid number.")

    comp_name = names[choice - 1]
    comp_id = int(static_mapping[comp_name])
    print(f"✅ Selected competition: {comp_name} (ID: {comp_id})")

    seasons_map = comp_utils.get_seasons_for_competition(comp_id)
    season_names = list(seasons_map.keys())
    if not season_names:
        print("No seasons found for this competition in competitions.json")
        return comp_name, comp_id, ["all"]

    print(f"\n📅 Available Seasons for {comp_name}:")
    print("-" * 50)
    print(" 0. All seasons")
    for idx, s in enumerate(season_names, 1):
        print(f"{idx:2d}. {s}")

    while True:
        ans = input(f"\n🎯 Select season (0 for all, 1-{len(season_names)}): ").strip()
        try:
            s_choice = int(ans)
            if s_choice == 0:
                return comp_name, comp_id, ["all"]
            if 1 <= s_choice <= len(season_names):
                return comp_name, comp_id, [season_names[s_choice - 1]]
        except Exception:
            pass
        print("Please enter a valid number.")


def confirm(prompt: str) -> bool:
    ans = input(f"{prompt} (y/n): ").strip().lower()
    return ans in ("", "y", "yes")


def run_step_scrape(comp_id: int, seasons: List[str]) -> Dict[str, any]:
    # Decide season parameter for scraper (None => all seasons)
    season_param = None if seasons == ["all"] else seasons[0]
    scraper = data_prep.StatsBombDataScraper(competition=comp_id, season=season_param)
    print("\n🚀 Scraping matches from StatsBomb ...")
    results = scraper.scrape_all_matches(output_dir=None)
    if results.get("success"):
        print("✅ Scraping finished.")
    else:
        print("❌ Scraping failed.")
    return results


def run_step_transform(comp_name: str, comp_id: int, seasons: List[str]) -> Dict[str, any]:
    print("\n🔄 Transforming Event-Level to Player-Level ...")
    results = player_transformer.transform_selected_data(comp_name, comp_id, seasons)
    if results.get("success"):
        print("✅ Transformation finished.")
    else:
        print("❌ Transformation failed.")
    return results


def run_step_aggregate(comp_name: str, comp_id: int, seasons: List[str]) -> Dict[str, any]:
    print("\n📊 Aggregating Player-Level into Statistics ...")
    aggregator = statistical_agg.StatisticalAggregator()
    results = aggregator.aggregate_selected_data(comp_name, comp_id, seasons)
    if results.get("success"):
        print("✅ Aggregation finished.")
    else:
        print("❌ Aggregation failed.")
    return results


def main():
    print("⚽ Soccer Analytics Pipeline - Main Orchestration")
    print("=" * 60)

    comp_name, comp_id, seasons = prompt_competition_and_season()

    # Step 1: Scrape
    if confirm("Run Step 1 - Scrape data now?"):
        scrape_res = run_step_scrape(comp_id, seasons)
        if not scrape_res.get("success"):
            print("Stopping pipeline due to scraping failure.")
            return

    # Step 2: Transform
    if confirm("Run Step 2 - Transform to player-level?"):
        transform_res = run_step_transform(comp_name, comp_id, seasons)
        if not transform_res.get("success"):
            print("Stopping pipeline due to transformation failure.")
            return

    # Step 3: Aggregate
    if confirm("Run Step 3 - Aggregate statistics?"):
        agg_res = run_step_aggregate(comp_name, comp_id, seasons)
        if not agg_res.get("success"):
            print("Stopping pipeline due to aggregation failure.")
            return

    print("\n🎉 Pipeline steps completed.")


if __name__ == "__main__":
    main()