"""
Data Scraping Module for Soccer Analytics Pipeline

This module handles data acquisition from StatsBomb API for FA Women's Super League.
It provides functionality to fetch match data, event data, and organize them into
structured datasets for further analysis.

Author: Michael Xu
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import importlib.util
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from statsbombpy import sb
except ImportError:
    print("Warning: statsbombpy not installed. Please install with: pip install statsbombpy")
    sb = None

# Load competitions_utils module directly
try:
    import Pipeline.utils.competitions_utils as comp_utils
except ImportError:
        # Fallback: import directly from file path
        this_dir = Path(__file__).parent
        file_path = this_dir / "utils" / "competitions_utils.py"
        if file_path.exists():
            spec = importlib.util.spec_from_file_location("competitions_utils", str(file_path))
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            comp_utils = module
        else:
            raise ImportError("competitions_utils module not found")


class StatsBombDataScraper:
    """
    Handles data scraping from StatsBomb API for FA Women's Super League
    """
    
    def __init__(self, competition: Union[int, str, None] = None, season: Union[int, str, None] = None):
        """Initialize the StatsBomb data scraper
        
        Args:
            competition: competition_id (int) or competition_name (str). Defaults to FA WSL if None.
            season: season_id (int) or season_name (str). If None, all available seasons for the competition.
        """
        if sb is None:
            raise ImportError("statsbombpy is required for data scraping. Install with: pip install statsbombpy")
        
        # Use competitions utilities
        resolve_competition_id = comp_utils.resolve_competition_id
        resolve_season_id = comp_utils.resolve_season_id

        # Default competition: FA Women's Super League when not provided
        resolved_competition_id = resolve_competition_id(competition) if competition is not None else 37
        self.competition_id = resolved_competition_id if resolved_competition_id is not None else 37

        # Optional season; if not provided, we'll fetch across all seasons
        self.season_id = resolve_season_id(self.competition_id, season) if season is not None else None
        self.matches_data = None
        self.matches_ids = None
    
    def get_competition_seasons(self) -> List[int]:
        """
        Get all available season IDs for FA Women's Super League
        
        Returns:
            List[int]: List of season IDs
        """
        print(f"🔍 Fetching available seasons for competition {self.competition_id}...")
        
        try:
            competitions = sb.competitions()
            season_ids = competitions[competitions['competition_id'] == self.competition_id]['season_id'].values
            print(f"✅ Found {len(season_ids)} seasons: {season_ids}")
            return season_ids.tolist()
        
        except Exception as e:
            print(f"❌ Error fetching seasons: {e}")
            return []
    
    def get_all_match_ids(self) -> List[int]:
        """
        Get all match IDs for FA Women's Super League across all seasons
        
        Returns:
            List[int]: List of all match IDs
        """
        print("🔍 Fetching match IDs...")
        
        season_ids = [self.season_id] if self.season_id is not None else self.get_competition_seasons()
        matches_ids = []
        
        for season_id in season_ids:
            try:
                matches = sb.matches(competition_id=self.competition_id, season_id=season_id)
                season_matches = matches['match_id'].values
                matches_ids.extend(season_matches)
                print(f"  📊 Season {season_id}: {len(season_matches)} matches")
            
            except Exception as e:
                print(f"  ⚠️ Error fetching matches for season {season_id}: {e}")
                continue
        
        self.matches_ids = matches_ids
        print(f"✅ Total matches found: {len(matches_ids)}")
        return matches_ids
    
    def fetch_match_events(self, match_id: int) -> pd.DataFrame:
        """
        Fetch event data for a specific match
        
        Args:
            match_id (int): The match ID to fetch events for
            
        Returns:
            pd.DataFrame: Event data for the match
        """
        try:
            events = sb.events(match_id=match_id)
            return events
        
        except Exception as e:
            print(f"❌ Error fetching events for match {match_id}: {e}")
            return pd.DataFrame()
    
    def save_match_data(self, match_id: int, events_df: pd.DataFrame, 
                       output_dir: Path) -> Tuple[bool, str]:
        """
        Save match events data to CSV file
        
        Args:
            match_id (int): The match ID
            events_df (pd.DataFrame): Event data to save
            output_dir (Path): Output directory path
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Create output directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save events data
            events_file = output_dir / f"match_{match_id}.csv"
            events_df.to_csv(events_file, index=False)
            
            return True, f"Events saved: {events_file} ({len(events_df)} events)"
        
        except Exception as e:
            return False, f"Error saving match {match_id}: {e}"
    
    def scrape_all_matches(self, output_dir: Optional[str] = None, 
                          progress_callback=None) -> Dict[str, any]:
        """
        Scrape all matches and save event data
        
        Args:
            output_dir (str): Directory to save event data. If None, defaults to per-competition/season folder.
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            Dict[str, any]: Scraping results and statistics
        """
        print("🚀 Starting data scraping...")
        print("=" * 70)
        
        # Get all match IDs
        if self.matches_ids is None:
            self.get_all_match_ids()
        
        if not self.matches_ids:
            return {"success": False, "error": "No match IDs found"}
        
        # Setup base competition directory
        WORKSPACE_ROOT = comp_utils.WORKSPACE_ROOT
        comp_folder = str(self.competition_id)
        # Try to get a friendly folder name from competitions file
        name_map = comp_utils.get_competition_id_mapping()
        # reverse map id -> name (first match)
        comp_name = next((name for name, cid in name_map.items() if int(cid) == int(self.competition_id)), str(self.competition_id))
        safe_name = comp_name.replace(' ', '_')
        comp_folder = safe_name if safe_name else comp_folder
        
        # If specific season, scrape only that season
        if self.season_id is not None:
            return self._scrape_single_season(comp_folder, output_dir, progress_callback)
        else:
            # Scrape all seasons with organized folders
            return self._scrape_all_seasons_organized(comp_folder, output_dir, progress_callback)
    
    def _scrape_single_season(self, comp_folder: str, output_dir: Optional[str], progress_callback) -> Dict[str, any]:
        """Scrape matches for a single season"""
        # Get season name from competitions data
        seasons_map = comp_utils.get_seasons_for_competition(self.competition_id)
        season_name = next((name for name, sid in seasons_map.items() if int(sid) == int(self.season_id)), str(self.season_id))
        
        # Create safe folder name - convert / to - and spaces to _
        safe_season = season_name.replace('/', '-').replace(' ', '_')
        
        if output_dir is None:
            output_path = comp_utils.WORKSPACE_ROOT / "Match" / comp_folder / safe_season / "Event-Level"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        return self._process_matches(output_path, season_name, progress_callback)
    
    def _scrape_all_seasons_organized(self, comp_folder: str, output_dir: Optional[str], progress_callback) -> Dict[str, any]:
        """Scrape matches for all seasons, organizing each season in its own folder"""
        print(f"📅 Scraping all available seasons for {comp_folder}")
        
        # Get all available seasons for this competition
        seasons_map = comp_utils.get_seasons_for_competition(self.competition_id)
        if not seasons_map:
            return {"success": False, "error": "No seasons found for this competition"}
        
        total_stats = {
            "total_matches": 0,
            "successful": 0,
            "failed": 0,
            "total_events": 0,
            "errors": [],
            "seasons_processed": 0
        }
        
        # Process each season separately
        for season_name, season_id in seasons_map.items():
            print(f"\n📅 Processing season: {season_name}")
            print("-" * 50)
            
            # Create season-specific folder
            # Create safe folder name - convert / to - and spaces to _
            safe_season = season_name.replace('/', '-').replace(' ', '_')
            season_output_path = comp_utils.WORKSPACE_ROOT / "Match" / comp_folder / safe_season / "Event-Level"
            season_output_path.mkdir(parents=True, exist_ok=True)
            
            # Get matches for this specific season
            try:
                matches = sb.matches(competition_id=self.competition_id, season_id=season_id)
                season_match_ids = matches['match_id'].values.tolist()
                
                if not season_match_ids:
                    print(f"  ⚠️ No matches found for season {season_name}")
                    continue
                
                print(f"  📊 Found {len(season_match_ids)} matches for {season_name}")
                
                # Process matches for this season
                season_stats = self._process_matches_for_season(season_match_ids, season_output_path, season_name)
                
                # Aggregate stats
                total_stats["total_matches"] += season_stats["total_matches"]
                total_stats["successful"] += season_stats["successful"]
                total_stats["failed"] += season_stats["failed"]
                total_stats["total_events"] += season_stats["total_events"]
                total_stats["errors"].extend(season_stats["errors"])
                total_stats["seasons_processed"] += 1
                
            except Exception as e:
                error_msg = f"Error processing season {season_name}: {e}"
                print(f"  ❌ {error_msg}")
                total_stats["errors"].append(error_msg)
                continue
        
        # Final statistics
        print("\n" + "=" * 70)
        print("📊 ALL SEASONS SCRAPING COMPLETE")
        print("=" * 70)
        print(f"✅ Seasons processed: {total_stats['seasons_processed']}")
        print(f"✅ Successful matches: {total_stats['successful']}")
        print(f"❌ Failed matches: {total_stats['failed']}")
        print(f"📈 Total events collected: {total_stats['total_events']:,}")
        print(f"📁 Data organized in: Match/{comp_folder}/<Season>/Event-Level/")
        
        if total_stats["errors"]:
            print(f"\n⚠️ Errors encountered: {len(total_stats['errors'])}")
            for error in total_stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        total_stats["success"] = total_stats["successful"] > 0
        return total_stats
    
    def _process_matches_for_season(self, match_ids: List[int], output_path: Path, season_name: str) -> Dict[str, any]:
        """Process matches for a specific season"""
        stats = {
            "total_matches": len(match_ids),
            "successful": 0,
            "failed": 0,
            "total_events": 0,
            "errors": []
        }
        
        for i, match_id in enumerate(match_ids):
            print(f"  Processing match {i+1}/{len(match_ids)}: {match_id}")
            
            try:
                # Fetch events data
                events_df = self.fetch_match_events(match_id)
                
                if events_df.empty:
                    print(f"    ⚠️ No events data for match {match_id}")
                    stats["failed"] += 1
                    continue
                
                # Save data
                success, message = self.save_match_data(match_id, events_df, output_path)
                
                if success:
                    print(f"    ✅ {message}")
                    stats["successful"] += 1
                    stats["total_events"] += len(events_df)
                else:
                    print(f"    ❌ {message}")
                    stats["failed"] += 1
                    stats["errors"].append(message)
            
            except Exception as e:
                error_msg = f"Unexpected error processing match {match_id}: {e}"
                print(f"    ❌ {error_msg}")
                stats["failed"] += 1
                stats["errors"].append(error_msg)
                continue
        
        return stats
    
    def _process_matches(self, output_path: Path, season_name: str, progress_callback) -> Dict[str, any]:
        """Process all matches and save to output path"""
        stats = {
            "total_matches": len(self.matches_ids),
            "successful": 0,
            "failed": 0,
            "total_events": 0,
            "errors": []
        }
        
        print(f"📁 Output directory: {output_path}")
        print(f"📅 Scraping season: {season_name}")
        print(f"📊 Processing {len(self.matches_ids)} matches...")
        print("=" * 70)
        
        # Process each match
        for i, match_id in enumerate(self.matches_ids):
            print(f"Processing match {i+1}/{len(self.matches_ids)}: {match_id}")
            
            try:
                # Fetch events data
                events_df = self.fetch_match_events(match_id)
                
                if events_df.empty:
                    print(f"  ⚠️ No events data for match {match_id}")
                    stats["failed"] += 1
                    continue
                
                # Save data
                success, message = self.save_match_data(match_id, events_df, output_path)
                
                if success:
                    print(f"  ✅ {message}")
                    stats["successful"] += 1
                    stats["total_events"] += len(events_df)
                else:
                    print(f"  ❌ {message}")
                    stats["failed"] += 1
                    stats["errors"].append(message)
                
                # Progress callback
                if progress_callback:
                    progress_callback(i + 1, len(self.matches_ids), match_id, success)
            
            except Exception as e:
                error_msg = f"Unexpected error processing match {match_id}: {e}"
                print(f"  ❌ {error_msg}")
                stats["failed"] += 1
                stats["errors"].append(error_msg)
                continue
            
            print("-" * 70)
        
        # Final statistics
        print("\n" + "=" * 70)
        print("📊 SCRAPING COMPLETE")
        print("=" * 70)
        print(f"✅ Successful matches: {stats['successful']}")
        print(f"❌ Failed matches: {stats['failed']}")
        print(f"📈 Total events collected: {stats['total_events']:,}")
        print(f"📁 Data saved to: {output_path}")
        
        if stats["errors"]:
            print(f"\n⚠️ Errors encountered: {len(stats['errors'])}")
            for error in stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        stats["success"] = stats["successful"] > 0
        return stats


def interactive_competition_scrape() -> Dict[str, any]:
    """
    Interactive helper: list competitions, let user choose by number, optional season,
    then scrape events into Match/<Competition Name>/Event-Level.
    """
    # Use the exact order from the static mapping constant loaded in competitions_utils
    static_mapping = getattr(comp_utils, "COMPETITION_ID_MAPPING", {})
    if not static_mapping:
        # Fallback to dynamic mapping if static not available
        mapping = comp_utils.get_competition_id_mapping()
        if not mapping:
            print("❌ No competitions mapping available")
            return {"success": False, "error": "No competitions mapping"}
        names = list(mapping.keys())
    else:
        names = list(static_mapping.keys())
    print("\nAvailable competitions:")
    for idx, name in enumerate(names, start=1):
        print(f"{idx}. {name}")

    choice = input("\nSelect competition you want to scrape: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(names)):
        return {"success": False, "error": "Invalid selection"}

    comp_name = names[int(choice) - 1]
    # Use static mapping for id resolution if present, else dynamic
    if static_mapping:
        comp_id = static_mapping[comp_name]
    else:
        comp_id = comp_utils.get_competition_id_mapping()[comp_name]

    # Optional season selection
    seasons_map = comp_utils.get_seasons_for_competition(comp_name)
    season_choice: Optional[Union[int, str]] = None
    if seasons_map:
        season_names = sorted(seasons_map.keys())
        print("\nAvailable seasons:")
        for idx, sname in enumerate(season_names, start=1):
            print(f"{idx}. {sname}")
        s_in = input("Enter season number (blank for all): ").strip()
        if s_in.isdigit() and 1 <= int(s_in) <= len(season_names):
            season_choice = season_names[int(s_in) - 1]

    scraper = StatsBombDataScraper(competition=comp_name, season=season_choice)
    return scraper.scrape_all_matches()


def main():
    """
    Main function to run data scraping
    """
    print("⚽ Soccer Analytics Pipeline - Data Scraping Module")
    print("=" * 60)
    
    try:
        
        # Run scraping
        results = interactive_competition_scrape()
        
        if results["success"]:
            print(f"\n🎉 Data scraping completed successfully!")
            print(f"📊 Collected data for {results['successful']} matches")
        else:
            print(f"\n❌ Data scraping failed!")
            
    except Exception as e:
        print(f"❌ Fatal error in data scraping: {e}")


if __name__ == "__main__":
    main()
