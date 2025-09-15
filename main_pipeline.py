"""
Main Soccer Analytics Pipeline Orchestration Script

This script orchestrates the entire soccer analytics pipeline from data scraping
to recruitment analysis and visualization. It provides a comprehensive workflow
for processing FA Women's Super League data and generating insights.

Author: Michael Xu
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import pipeline modules
# Note: Using importlib to handle numbered module names
import importlib.util
import sys
from pathlib import Path

# Get current directory for relative imports
current_dir = Path(__file__).parent

# Import numbered modules using importlib
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, current_dir / file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load all pipeline modules
data_prep = load_module("data_preparation", "01_data_preparation.py")
player_transformer = load_module("player_transformer", "02_player_level_transformer.py") 
statistical_agg = load_module("statistical_aggregator", "03_statistical_aggregator.py")
recruitment_analyzer = load_module("recruitment_analyzer", "04_recruitment_analyzer.py")
visualization = load_module("recruitment_tools", "05_recruitment_tools.py")

# Extract classes and functions for easier use
StatsBombDataScraper = data_prep.StatsBombDataScraper
transform_to_player_level = player_transformer.transform_to_player_level
StatisticalAggregator = statistical_agg.StatisticalAggregator
RecruitmentAnalyzer = recruitment_analyzer.RecruitmentAnalyzer
SoccerRecruitmentTools = visualization.SoccerRecruitmentTools


class SoccerAnalyticsPipeline:
    """
    Main pipeline orchestrator for soccer analytics
    """
    
    def __init__(self, base_dir: str = "."):
        """
        Initialize the soccer analytics pipeline
        
        Args:
            base_dir (str): Base directory for the pipeline
        """
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
        # Initialize components
        self.scraper = None
        self.aggregator = StatisticalAggregator()
        self.recruitment_analyzer = None
        self.visualization_tools = SoccerVisualizationTools()
        
        # Data storage
        self.raw_data = {}
        self.processed_data = {}
        self.statistics = {}
        self.insights = {}
    
    def setup_directories(self):
        """Setup required directories for the pipeline"""
        directories = [
            "Match/Event-Level",
            "Match/Player-Level", 
            "Statistics",
            "Visualizations",
            "Recruitment_Analysis"
        ]
        
        for directory in directories:
            dir_path = self.base_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {dir_path}")
    
    def run_data_scraping(self, force_refresh: bool = False) -> Dict[str, any]:
        """
        Run data scraping from StatsBomb API
        
        Args:
            force_refresh (bool): Whether to force refresh existing data
            
        Returns:
            Dict[str, any]: Scraping results
        """
        print("🚀 STEP 1: DATA SCRAPING")
        print("=" * 50)
        
        event_dir = self.base_dir / "Match/Event-Level"
        
        # Check if data already exists
        if not force_refresh and event_dir.exists() and len(list(event_dir.glob("*.csv"))) > 0:
            print(f"📊 Event data already exists in {event_dir}")
            print("   Use force_refresh=True to re-scrape data")
            
            # Count existing files
            existing_files = list(event_dir.glob("*.csv"))
            return {
                "success": True,
                "message": f"Found {len(existing_files)} existing event files",
                "existing_files": len(existing_files)
            }
        
        try:
            # Initialize scraper
            self.scraper = StatsBombDataScraper()
            
            # Run scraping
            results = self.scraper.scrape_all_matches(str(event_dir))
            
            if results["success"]:
                print(f"✅ Data scraping completed successfully!")
                print(f"📊 Collected data for {results['successful']} matches")
                self.raw_data["scraping_results"] = results
            else:
                print(f"❌ Data scraping failed!")
                
            return results
            
        except Exception as e:
            print(f"❌ Error in data scraping: {e}")
            return {"success": False, "error": str(e)}
    
    def run_player_level_transformation(self, force_refresh: bool = False) -> Dict[str, any]:
        """
        Transform event-level data to player-level statistics
        
        Args:
            force_refresh (bool): Whether to force refresh existing data
            
        Returns:
            Dict[str, any]: Transformation results
        """
        print("\n🔄 STEP 2: PLAYER LEVEL TRANSFORMATION")
        print("=" * 50)
        
        event_dir = self.base_dir / "Match/Event-Level"
        player_dir = self.base_dir / "Match/Player-Level"
        
        # Check if transformation already done
        if not force_refresh and player_dir.exists() and len(list(player_dir.glob("*.csv"))) > 0:
            print(f"📊 Player data already exists in {player_dir}")
            print("   Use force_refresh=True to re-transform data")
            
            existing_files = list(player_dir.glob("*.csv"))
            return {
                "success": True,
                "message": f"Found {len(existing_files)} existing player files",
                "existing_files": len(existing_files)
            }
        
        try:
            # Get all event files
            event_files = list(event_dir.glob("*.csv"))
            
            if not event_files:
                return {"success": False, "error": "No event files found. Run data scraping first."}
            
            print(f"📊 Processing {len(event_files)} match files...")
            
            successful_transformations = 0
            failed_transformations = 0
            
            for i, event_file in enumerate(event_files):
                match_id = event_file.stem.replace("match_", "")
                print(f"Processing match {i+1}/{len(event_files)}: {match_id}")
                
                try:
                    # Load event data
                    events_df = pd.read_csv(event_file)
                    
                    # Transform to player level
                    analyzer, player_stats = transform_to_player_level(events_df)
                    
                    # Save player statistics
                    player_file = player_dir / f"match_{match_id}.csv"
                    player_stats.to_csv(player_file, index=False)
                    
                    print(f"  ✅ Player stats saved: {len(player_stats)} players")
                    successful_transformations += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing match {match_id}: {e}")
                    failed_transformations += 1
                    continue
                
                print("-" * 50)
            
            results = {
                "success": successful_transformations > 0,
                "successful_transformations": successful_transformations,
                "failed_transformations": failed_transformations,
                "total_files": len(event_files)
            }
            
            print(f"\n✅ Player level transformation completed!")
            print(f"📊 Successful: {successful_transformations}")
            print(f"❌ Failed: {failed_transformations}")
            
            self.processed_data["transformation_results"] = results
            return results
            
        except Exception as e:
            print(f"❌ Error in player level transformation: {e}")
            return {"success": False, "error": str(e)}
    
    def run_statistical_aggregation(self, force_refresh: bool = False) -> Dict[str, any]:
        """
        Create statistical datasets from player-level data
        
        Args:
            force_refresh (bool): Whether to force refresh existing data
            
        Returns:
            Dict[str, any]: Aggregation results
        """
        print("\n📊 STEP 3: STATISTICAL AGGREGATION")
        print("=" * 50)
        
        player_dir = self.base_dir / "Match/Player-Level"
        stats_dir = self.base_dir / "Statistics"
        
        # Check if statistics already exist
        if not force_refresh and stats_dir.exists():
            existing_stats = list(stats_dir.glob("statistics_*.csv"))
            if existing_stats:
                print(f"📊 Statistics already exist in {stats_dir}")
                print("   Use force_refresh=True to re-aggregate data")
                
                # Load existing statistics
                self.statistics = {}
                for stat_file in existing_stats:
                    dataset_name = stat_file.stem.replace("statistics_", "")
                    self.statistics[dataset_name] = pd.read_csv(stat_file)
                    print(f"  📈 Loaded {dataset_name}: {len(self.statistics[dataset_name])} players")
                
                return {
                    "success": True,
                    "message": f"Found {len(existing_stats)} existing statistical datasets",
                    "existing_datasets": len(existing_stats)
                }
        
        try:
            # Run aggregation
            results = self.aggregator.aggregate_player_stats(
                str(player_dir), 
                str(stats_dir)
            )
            
            if results["success"]:
                print(f"✅ Statistical aggregation completed successfully!")
                print(f"📊 Created datasets for {results['unique_players']} players")
                
                # Load the created datasets
                self.statistics = results["datasets"]
                
            else:
                print(f"❌ Statistical aggregation failed!")
                
            return results
            
        except Exception as e:
            print(f"❌ Error in statistical aggregation: {e}")
            return {"success": False, "error": str(e)}
    
    def run_recruitment_analysis(self) -> Dict[str, any]:
        """
        Run recruitment analysis and generate insights
        
        Returns:
            Dict[str, any]: Recruitment analysis results
        """
        print("\n🎯 STEP 4: RECRUITMENT ANALYSIS")
        print("=" * 50)
        
        try:
            # Check if required datasets exist
            if 'performance_scores' not in self.statistics or 'totals' not in self.statistics:
                return {"success": False, "error": "Required datasets not found. Run statistical aggregation first."}
            
            # Initialize recruitment analyzer
            self.recruitment_analyzer = RecruitmentAnalyzer(
                self.statistics['performance_scores'],
                self.statistics['totals']
            )
            
            # Generate recruitment insights
            insights = self.recruitment_analyzer.get_recruitment_insights()
            
            # Export recruitment targets
            recruitment_file = self.base_dir / "Recruitment_Analysis/recruitment_targets.csv"
            recruitment_file.parent.mkdir(exist_ok=True)
            
            exported_file = self.recruitment_analyzer.export_recruitment_targets(str(recruitment_file))
            
            self.insights = insights
            
            print(f"✅ Recruitment analysis completed successfully!")
            print(f"📊 Analyzed {insights['total_players']} players")
            print(f"🎯 Identified {len(insights['hidden_gems_players'])} hidden gems")
            print(f"📈 Exported targets to: {exported_file}")
            
            return {
                "success": True,
                "insights": insights,
                "exported_file": exported_file
            }
            
        except Exception as e:
            print(f"❌ Error in recruitment analysis: {e}")
            return {"success": False, "error": str(e)}
    
    def run_visualization(self, save_plots: bool = True) -> Dict[str, any]:
        """
        Generate visualizations and dashboards
        
        Args:
            save_plots (bool): Whether to save plots to files
            
        Returns:
            Dict[str, any]: Visualization results
        """
        print("\n📊 STEP 5: VISUALIZATION")
        print("=" * 50)
        
        try:
            # Check if required data exists
            if not self.recruitment_analyzer or not self.statistics:
                return {"success": False, "error": "Required data not found. Run previous steps first."}
            
            viz_dir = self.base_dir / "Visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            results = {"plots_created": [], "errors": []}
            
            # 1. Performance distribution plot
            try:
                save_path = str(viz_dir / "performance_distribution.png") if save_plots else None
                self.visualization_tools.plot_performance_distribution(
                    self.recruitment_analyzer.recruitment_df, save_path
                )
                results["plots_created"].append("performance_distribution")
            except Exception as e:
                results["errors"].append(f"Performance distribution: {e}")
            
            # 2. Top performers plot
            try:
                save_path = str(viz_dir / "top_performers.png") if save_plots else None
                self.visualization_tools.plot_top_performers(
                    self.statistics['totals'], save_path
                )
                results["plots_created"].append("top_performers")
            except Exception as e:
                results["errors"].append(f"Top performers: {e}")
            
            # 3. Per-90 comparison plot
            try:
                save_path = str(viz_dir / "per_90_comparison.png") if save_plots else None
                self.visualization_tools.plot_per_90_comparison(
                    self.statistics['per_90'], save_path
                )
                results["plots_created"].append("per_90_comparison")
            except Exception as e:
                results["errors"].append(f"Per-90 comparison: {e}")
            
            # 4. Specialist analysis plots
            try:
                if 'insights' in self.insights and 'specialists' in self.insights:
                    save_path = str(viz_dir / "specialist_analysis.png") if save_plots else None
                    self.visualization_tools.create_specialist_analysis_plots(
                        self.insights['specialists'], save_path
                    )
                    results["plots_created"].append("specialist_analysis")
            except Exception as e:
                results["errors"].append(f"Specialist analysis: {e}")
            
            # 5. Example comparison dashboard
            try:
                if len(self.recruitment_analyzer.recruitment_df) > 0:
                    # Get top 4 players for comparison
                    top_players = self.recruitment_analyzer.recruitment_df.nlargest(4, 'performance_score')['player_name'].tolist()
                    
                    save_path = str(viz_dir / "player_comparison_dashboard.png") if save_plots else None
                    self.visualization_tools.create_enhanced_comparison_dashboard(
                        self.recruitment_analyzer.recruitment_df, top_players, save_path
                    )
                    results["plots_created"].append("player_comparison_dashboard")
            except Exception as e:
                results["errors"].append(f"Comparison dashboard: {e}")
            
            print(f"✅ Visualization completed!")
            print(f"📊 Created {len(results['plots_created'])} plots")
            if results["errors"]:
                print(f"⚠️ {len(results['errors'])} errors encountered")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in visualization: {e}")
            return {"success": False, "error": str(e)}
    
    def run_full_pipeline(self, force_refresh: bool = False, save_plots: bool = True) -> Dict[str, any]:
        """
        Run the complete soccer analytics pipeline
        
        Args:
            force_refresh (bool): Whether to force refresh existing data
            save_plots (bool): Whether to save visualization plots
            
        Returns:
            Dict[str, any]: Complete pipeline results
        """
        print("⚽ SOCCER ANALYTICS PIPELINE - FULL EXECUTION")
        print("=" * 70)
        print("Author: Michael Xu")
        print("Framework: PSCDL Performance Scoring System")
        print("=" * 70)
        
        pipeline_results = {
            "data_scraping": None,
            "player_transformation": None,
            "statistical_aggregation": None,
            "recruitment_analysis": None,
            "visualization": None,
            "overall_success": False
        }
        
        try:
            # Step 1: Data Scraping
            scraping_results = self.run_data_scraping(force_refresh)
            pipeline_results["data_scraping"] = scraping_results
            
            if not scraping_results["success"] and not scraping_results.get("existing_files", 0):
                print("❌ Pipeline failed at data scraping step")
                return pipeline_results
            
            # Step 2: Player Level Transformation
            transformation_results = self.run_player_level_transformation(force_refresh)
            pipeline_results["player_transformation"] = transformation_results
            
            if not transformation_results["success"] and not transformation_results.get("existing_files", 0):
                print("❌ Pipeline failed at player transformation step")
                return pipeline_results
            
            # Step 3: Statistical Aggregation
            aggregation_results = self.run_statistical_aggregation(force_refresh)
            pipeline_results["statistical_aggregation"] = aggregation_results
            
            if not aggregation_results["success"] and not aggregation_results.get("existing_datasets", 0):
                print("❌ Pipeline failed at statistical aggregation step")
                return pipeline_results
            
            # Step 4: Recruitment Analysis
            recruitment_results = self.run_recruitment_analysis()
            pipeline_results["recruitment_analysis"] = recruitment_results
            
            if not recruitment_results["success"]:
                print("❌ Pipeline failed at recruitment analysis step")
                return pipeline_results
            
            # Step 5: Visualization
            visualization_results = self.run_visualization(save_plots)
            pipeline_results["visualization"] = visualization_results
            
            # Overall success
            pipeline_results["overall_success"] = True
            
            print("\n" + "=" * 70)
            print("🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("📊 Generated comprehensive soccer analytics insights")
            print("🎯 Created recruitment recommendations")
            print("📈 Produced visualization dashboards")
            print("💾 All data saved to organized directories")
            
            return pipeline_results
            
        except Exception as e:
            print(f"❌ Fatal error in pipeline execution: {e}")
            pipeline_results["fatal_error"] = str(e)
            return pipeline_results
    
    def get_pipeline_summary(self) -> Dict[str, any]:
        """
        Get summary of pipeline execution and results
        
        Returns:
            Dict[str, any]: Pipeline summary
        """
        summary = {
            "base_directory": str(self.base_dir),
            "directories": {
                "event_data": str(self.base_dir / "Match/Event-Level"),
                "player_data": str(self.base_dir / "Match/Player-Level"),
                "statistics": str(self.base_dir / "Statistics"),
                "visualizations": str(self.base_dir / "Visualizations"),
                "recruitment": str(self.base_dir / "Recruitment_Analysis")
            },
            "data_available": {
                "event_files": len(list((self.base_dir / "Match/Event-Level").glob("*.csv"))) if (self.base_dir / "Match/Event-Level").exists() else 0,
                "player_files": len(list((self.base_dir / "Match/Player-Level").glob("*.csv"))) if (self.base_dir / "Match/Player-Level").exists() else 0,
                "statistics_datasets": len(list((self.base_dir / "Statistics").glob("*.csv"))) if (self.base_dir / "Statistics").exists() else 0,
                "visualization_files": len(list((self.base_dir / "Visualizations").glob("*.png"))) if (self.base_dir / "Visualizations").exists() else 0
            },
            "statistics": self.statistics if self.statistics else {},
            "insights": self.insights if self.insights else {}
        }
        
        return summary


def main():
    """
    Main function to run the soccer analytics pipeline
    """
    print("⚽ Soccer Analytics Pipeline - Main Orchestration")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = SoccerAnalyticsPipeline()
    
    # Get user preferences
    print("\n🔧 Pipeline Configuration:")
    force_refresh = input("Force refresh existing data? (y/N): ").lower().startswith('y')
    save_plots = input("Save visualization plots? (Y/n): ").lower() != 'n'
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(force_refresh=force_refresh, save_plots=save_plots)
    
    # Display summary
    if results["overall_success"]:
        print("\n📋 PIPELINE SUMMARY:")
        summary = pipeline.get_pipeline_summary()
        
        print(f"📁 Base Directory: {summary['base_directory']}")
        print(f"📊 Event Files: {summary['data_available']['event_files']}")
        print(f"👥 Player Files: {summary['data_available']['player_files']}")
        print(f"📈 Statistics Datasets: {summary['data_available']['statistics_datasets']}")
        print(f"🎨 Visualization Files: {summary['data_available']['visualization_files']}")
        
        if summary['insights']:
            print(f"🎯 Total Players Analyzed: {summary['insights'].get('total_players', 'N/A')}")
            print(f"💎 Hidden Gems Identified: {len(summary['insights'].get('hidden_gems_players', []))}")
    else:
        print("\n❌ Pipeline execution failed. Check error messages above.")


#if __name__ == "__main__":
#    main()


