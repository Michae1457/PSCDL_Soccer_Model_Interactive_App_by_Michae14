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
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from statsbombpy import sb
except ImportError:
    print("Warning: statsbombpy not installed. Please install with: pip install statsbombpy")
    sb = None


class StatsBombDataScraper:
    """
    Handles data scraping from StatsBomb API for FA Women's Super League
    """
    
    def __init__(self):
        """Initialize the StatsBomb data scraper"""
        if sb is None:
            raise ImportError("statsbombpy is required for data scraping. Install with: pip install statsbombpy")
        
        self.competition_id = 37  # FA Women's Super League
        self.matches_data = None
        self.matches_ids = None
    
    def get_competition_seasons(self) -> List[int]:
        """
        Get all available season IDs for FA Women's Super League
        
        Returns:
            List[int]: List of season IDs
        """
        print("🔍 Fetching available seasons for FA Women's Super League...")
        
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
        print("🔍 Fetching all match IDs for FA Women's Super League...")
        
        season_ids = self.get_competition_seasons()
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
    
    def scrape_all_matches(self, output_dir: str = "Soccer Analytics Pipeline/Match/Event-Level", 
                          progress_callback=None) -> Dict[str, any]:
        """
        Scrape all matches and save event data
        
        Args:
            output_dir (str): Directory to save event data
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            Dict[str, any]: Scraping results and statistics
        """
        print("🚀 Starting data scraping for all FA Women's Super League matches...")
        print("=" * 70)
        
        # Get all match IDs
        if self.matches_ids is None:
            self.get_all_match_ids()
        
        if not self.matches_ids:
            return {"success": False, "error": "No match IDs found"}
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Scraping statistics
        stats = {
            "total_matches": len(self.matches_ids),
            "successful": 0,
            "failed": 0,
            "total_events": 0,
            "errors": []
        }
        
        print(f"📁 Output directory: {output_path}")
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


def main():
    """
    Main function to run data scraping
    """
    print("⚽ Soccer Analytics Pipeline - Data Scraping Module")
    print("=" * 60)
    
    try:
        # Initialize scraper
        scraper = StatsBombDataScraper()
        
        # Run scraping
        results = scraper.scrape_all_matches()
        
        if results["success"]:
            print(f"\n🎉 Data scraping completed successfully!")
            print(f"📊 Collected data for {results['successful']} matches")
        else:
            print(f"\n❌ Data scraping failed!")
            
    except Exception as e:
        print(f"❌ Fatal error in data scraping: {e}")


if __name__ == "__main__":
    main()
