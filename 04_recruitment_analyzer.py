"""
Recruitment Analysis Module

This module provides recruitment strategy tools and analysis based on the PSCDL framework.
It includes specialist identification, hidden gems detection, and recruitment insights
to help scouts and analysts make data-driven decisions.

Author: Michael Xu
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class RecruitmentAnalyzer:
    """
    Provides recruitment analysis tools based on PSCDL performance framework
    """
    
    def __init__(self, performance_df: pd.DataFrame, totals_df: pd.DataFrame):
        """
        Initialize the recruitment analyzer
        
        Args:
            performance_df (pd.DataFrame): Performance scores dataset
            totals_df (pd.DataFrame): Totals statistics dataset
        """
        self.performance_df = performance_df.copy()
        self.totals_df = totals_df.copy()
        self.recruitment_df = None
        self._prepare_recruitment_data()
    
    def _prepare_recruitment_data(self):
        """Prepare recruitment data by merging performance and totals datasets"""
        # Merge with totals for additional context
        self.recruitment_df = self.performance_df.merge(
            self.totals_df[['player_id', 'player_name', 'total_games', 'total_minutes_played']], 
            on=['player_id', 'player_name'], 
            how='left'
        )
        
        print(f"📊 Loaded recruitment data for {len(self.recruitment_df)} players")
        print(f"📈 Available metrics: {list(self.recruitment_df.columns)}")
    
    def get_performance_distribution(self) -> Dict[str, float]:
        """
        Get performance score distribution statistics
        
        Returns:
            Dict[str, float]: Performance distribution statistics
        """
        performance_scores = self.recruitment_df['performance_score']
        
        distribution = {
            "mean": performance_scores.mean(),
            "median": performance_scores.median(),
            "std": performance_scores.std(),
            "top_10_percent": performance_scores.quantile(0.9),
            "top_1_percent": performance_scores.quantile(0.99),
            "min": performance_scores.min(),
            "max": performance_scores.max()
        }
        
        return distribution
    
    def identify_specialists(self, min_games: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Identify top specialists in each PSCDL area
        
        Args:
            min_games (int): Minimum games required to be considered
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of specialists by area
        """
        # Filter players with sufficient game time
        qualified_players = self.recruitment_df[self.recruitment_df['total_games'] >= min_games].copy()
        
        specialists = {}
        
        # Define specialist areas and their labels
        specialist_areas = {
            'P_score': 'PASSING/CREATION',
            'S_score': 'FINISHING/SHOT VALUE', 
            'C_score': 'CARRYING/1V1',
            'D_score': 'DEFENDING/DISRUPTION'
        }
        
        print(f"\n🏆 IDENTIFYING TOP SPECIALISTS (≥{min_games} games)")
        print("=" * 70)
        
        # Find specialists in each area
        for area, label in specialist_areas.items():
            threshold = qualified_players[area].quantile(0.9)
            area_specialists = qualified_players[
                qualified_players[area] >= threshold
            ].sort_values(by=area, ascending=False)
            
            specialists[area] = area_specialists
            
            print(f"\n🏆 TOP {label} SPECIALISTS (≥{threshold:.1f}):")
            print("=" * 60)
            for i, (_, player) in enumerate(area_specialists.head(10).iterrows(), 1):
                print(f"{i:2d}. {player['player_name']:<25} | "
                      f"{area}: {player[area]:.1f} | "
                      f"Games: {player['total_games']:2d} | "
                      f"Overall: {player['performance_score']:.1f}")
        
        return specialists
    
    def find_hidden_gems(self, min_games: int = 5, overall_threshold_percentile: float = 0.7, 
                        specialist_threshold: float = 75.0) -> Tuple[List[Dict], List[str]]:
        """
        Find players with high potential in specific areas but lower overall scores
        
        Args:
            min_games (int): Minimum games required
            overall_threshold_percentile (float): Percentile threshold for overall performance
            specialist_threshold (float): Minimum specialist score threshold
            
        Returns:
            Tuple[List[Dict], List[str]]: (hidden_gems_list, hidden_gems_names)
        """
        qualified_players = self.recruitment_df[self.recruitment_df['total_games'] >= min_games].copy()
        
        # Define thresholds
        overall_threshold = qualified_players['performance_score'].quantile(overall_threshold_percentile)
        
        hidden_gems = []
        hidden_gems_players = []
        
        specialist_areas = {
            'P_score': 'PASSING/CREATION',
            'S_score': 'FINISHING/SHOT VALUE',
            'C_score': 'CARRYING/1V1', 
            'D_score': 'DEFENDING/DISRUPTION'
        }
        
        print(f"\n💎 FINDING HIDDEN GEMS")
        print("=" * 50)
        print("Players with high specialist potential but lower overall scores")
        print("(Potential for development and value recruitment)")
        
        for area, label in specialist_areas.items():
            # Find players with high specialist score but lower overall performance
            gems = qualified_players[
                (qualified_players[area] >= specialist_threshold) &
                (qualified_players['performance_score'] < overall_threshold)
            ].sort_values(by=area, ascending=False)
            
            if len(gems) > 0:
                print(f"\n💎 HIDDEN GEMS - {label} SPECIALISTS:")
                print("=" * 60)
                print("(High specialist potential but lower overall scores)")
                print("-" * 60)
                
                for i, (_, player) in enumerate(gems.head(8).iterrows(), 1):
                    print(f"{i:2d}. {player['player_name']:<25} | "
                          f"{area}: {player[area]:.1f} | "
                          f"Overall: {player['performance_score']:.1f} | "
                          f"Games: {player['total_games']:2d}")
                
                # Add top 5 gems from this area
                top_gems = gems.head(5).to_dict('records')
                hidden_gems.extend(top_gems)
                
                # Add player names to the list
                for gem in top_gems:
                    hidden_gems_players.append(gem['player_name'])
        
        # Remove duplicates while preserving order
        unique_hidden_gems = []
        seen_players = set()
        
        for gem in hidden_gems:
            if gem['player_name'] not in seen_players:
                unique_hidden_gems.append(gem)
                seen_players.add(gem['player_name'])
        
        return unique_hidden_gems, hidden_gems_players
    
    def calculate_radar_area(self, scores: List[float]) -> float:
        """
        Calculate the area of a radar chart polygon
        
        Args:
            scores (List[float]): List of 5 scores [P_score, S_score, C_score, D_score, L_score]
            
        Returns:
            float: Area of the radar chart polygon
        """
        # Convert scores to radians for polar coordinates
        # 5 axes evenly spaced around the circle (72 degrees apart)
        angles = [i * 2 * math.pi / 5 for i in range(5)]
        
        # Calculate area using the shoelace formula for polygon area
        # In polar coordinates: A = (1/2) * sum(r_i * r_{i+1} * sin(theta_{i+1} - theta_i))
        
        area = 0
        for i in range(5):
            j = (i + 1) % 5  # Next point (wrapping around)
            area += scores[i] * scores[j] * math.sin(angles[j] - angles[i])
        
        return abs(area) / 2
    
    def add_radar_area_metric(self, verbose=True) -> pd.DataFrame:
        """
        Add radar area metric to the recruitment dataframe
        
        Args:
            verbose (bool): Whether to print progress and results
            
        Returns:
            pd.DataFrame: Recruitment dataframe with radar area metrics
        """
        if verbose:
            print("🔍 CALCULATING RADAR CHART AREAS")
            print("=" * 50)
        
        # Calculate radar area for each player
        radar_areas = []
        for _, player in self.recruitment_df.iterrows():
            scores = [player['P_score'], player['S_score'], player['C_score'], 
                     player['D_score'], player['L_score']]
            area = self.calculate_radar_area(scores)
            radar_areas.append(area)
        
        self.recruitment_df['radar_area'] = radar_areas
        
        # Better normalization: scale to theoretical maximum (perfect player with all scores = 100)
        # For a regular pentagon with radius 100, area = (5/2) * 100^2 * sin(72°) ≈ 23,776
        theoretical_max_area = (5/2) * 100**2 * np.sin(np.radians(72))
        self.recruitment_df['radar_area_normalized'] = (self.recruitment_df['radar_area'] / theoretical_max_area) * 100
        
        # Also add percentile-based normalization for relative comparison
        self.recruitment_df['radar_area_percentile'] = self.recruitment_df['radar_area'].rank(pct=True) * 100
        
        # Display top players by radar area
        if verbose:
            print("\n📊 TOP PLAYERS BY RADAR AREA (Overall Ability Coverage)")
            print("=" * 80)
            top_radar_players = self.recruitment_df.nlargest(10, 'radar_area_normalized')[
                ['player_name', 'radar_area_normalized', 'radar_area_percentile', 'performance_score', 
                 'P_score', 'S_score', 'C_score', 'D_score', 'L_score']
            ]
            
            for i, (_, player) in enumerate(top_radar_players.iterrows(), 1):
                print(f"{i:2d}. {player['player_name']:<25} | "
                      f"Theoretical: {player['radar_area_normalized']:5.1f} | "
                      f"Percentile: {player['radar_area_percentile']:5.1f} | "
                      f"Performance: {player['performance_score']:5.1f} | "
                      f"P:{player['P_score']:4.1f} S:{player['S_score']:4.1f} C:{player['C_score']:4.1f} D:{player['D_score']:4.1f} L:{player['L_score']:4.1f}")

            print(f"\n💡 Interpretation:")
            print(f"   - Theoretical: 50 = half as good as perfect player")
            print(f"   - Percentile: 90 = better than 90% of players in dataset")
            print(f"   - Independent of performance score weighting")
        
        return self.recruitment_df
    
    def get_recruitment_insights(self) -> Dict[str, any]:
        """
        Generate comprehensive recruitment insights
        
        Returns:
            Dict[str, any]: Recruitment insights and recommendations
        """
        print("\n🎯 RECRUITMENT INSIGHTS & STRATEGIC RECOMMENDATIONS")
        print("=" * 70)
        
        # Performance distribution
        distribution = self.get_performance_distribution()
        
        # Identify specialists
        specialists = self.identify_specialists()
        
        # Find hidden gems
        hidden_gems, hidden_gems_players = self.find_hidden_gems()
        
        # Add radar area metrics
        self.add_radar_area_metric()
        
        # Get top all-rounders (high radar area)
        top_all_rounders = self.recruitment_df.nlargest(5, 'radar_area_normalized')
        
        insights = {
            "performance_distribution": distribution,
            "specialists": specialists,
            "hidden_gems": hidden_gems,
            "hidden_gems_players": hidden_gems_players,
            "top_all_rounders": top_all_rounders,
            "total_players": len(self.recruitment_df),
            "qualified_players": len(self.recruitment_df[self.recruitment_df['total_games'] >= 5])
        }
        
        return insights
    
    def get_player_comparison_data(self, player_names: List[str]) -> pd.DataFrame:
        """
        Get comparison data for specific players
        
        Args:
            player_names (List[str]): List of player names to compare
            
        Returns:
            pd.DataFrame: Comparison data for specified players
        """
        comparison_data = self.recruitment_df[self.recruitment_df['player_name'].isin(player_names)]
        
        if len(comparison_data) == 0:
            print("❌ No players found in the dataset")
            print("Available players:", self.recruitment_df['player_name'].head(10).tolist())
            return pd.DataFrame()
        
        if len(comparison_data) != len(player_names):
            found_players = comparison_data['player_name'].tolist()
            missing_players = [p for p in player_names if p not in found_players]
            print(f"⚠️ Some players not found: {missing_players}")
            print(f"✅ Comparing: {found_players}")
        
        return comparison_data
    
    def export_recruitment_targets(self, output_file: str = 'recruitment_targets.csv') -> str:
        """
        Export recruitment targets to CSV file
        
        Args:
            output_file (str): Output file path
            
        Returns:
            str: Path to exported file
        """
        # Create recruitment targets dataset
        targets = self.recruitment_df.copy()
        
        # Add recruitment categories
        qualified_players = targets[targets['total_games'] >= 5]
        
        # Elite performers (top 10% overall)
        elite_threshold = qualified_players['performance_score'].quantile(0.9)
        targets['is_elite'] = targets['performance_score'] >= elite_threshold
        
        # Specialists (top 10% in any area)
        specialist_areas = ['P_score', 'S_score', 'C_score', 'D_score']
        for area in specialist_areas:
            area_threshold = qualified_players[area].quantile(0.9)
            targets[f'is_{area.replace("_score", "_specialist")}'] = targets[area] >= area_threshold
        
        # All-rounders (top 25% radar area)
        radar_threshold = qualified_players['radar_area_normalized'].quantile(0.75)
        targets['is_all_rounder'] = targets['radar_area_normalized'] >= radar_threshold
        
        # Save to CSV
        targets.to_csv(output_file, index=False)
        
        print(f"✅ Exported recruitment targets to: {output_file}")
        print(f"📊 {len(targets)} players with recruitment categories")
        
        return output_file


def main():
    """
    Main function to demonstrate recruitment analysis with actual data
    """
    print("⚽ Soccer Analytics Pipeline - Recruitment Analysis Module")
    print("=" * 70)
    
    try:
        # Check if statistics files exist
        from pathlib import Path
        
        stats_dir = Path("Statistics")
        performance_file = stats_dir / "statistics_performance_scores.csv"
        totals_file = stats_dir / "statistics_totals.csv"
        
        if not performance_file.exists() or not totals_file.exists():
            print("❌ Required statistics files not found!")
            print("💡 Please run the statistical aggregation step first:")
            print("   python3 03_statistical_aggregator.py")
            print("\n📁 Looking for:")
            print(f"   • {performance_file}")
            print(f"   • {totals_file}")
            return
        
        print("📊 Loading statistics data...")
        
        # Load the datasets
        performance_df = pd.read_csv(performance_file)
        totals_df = pd.read_csv(totals_file)
        
        print(f"✅ Loaded performance data: {len(performance_df)} players")
        print(f"✅ Loaded totals data: {len(totals_df)} players")
        
        # Initialize recruitment analyzer
        print("\n🎯 Initializing Recruitment Analyzer...")
        analyzer = RecruitmentAnalyzer(performance_df, totals_df)
        
        # Generate comprehensive insights
        print("\n🔍 Generating Recruitment Insights...")
        insights = analyzer.get_recruitment_insights()
        
        # Display detailed results
        print("\n" + "=" * 70)
        print("📊 RECRUITMENT ANALYSIS RESULTS")
        print("=" * 70)
        
        # Performance distribution
        dist = insights['performance_distribution']
        print(f"\n🎯 PERFORMANCE SCORE DISTRIBUTION:")
        print(f"   • Mean: {dist['mean']:.2f}")
        print(f"   • Median: {dist['median']:.2f}")
        print(f"   • Top 10%: {dist['top_10_percent']:.2f}")
        print(f"   • Top 1%: {dist['top_1_percent']:.2f}")
        print(f"   • Range: {dist['min']:.1f} - {dist['max']:.1f}")
        
        # Export recruitment targets
        print(f"\n💾 EXPORTING RECRUITMENT TARGETS...")
        output_file = analyzer.export_recruitment_targets("recruitment_targets.csv")
        print(f"✅ Recruitment targets saved to: {output_file}")
        
        # Summary statistics
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Total Players Analyzed: {insights['total_players']}")
        print(f"   • Qualified Players (≥5 games): {insights['qualified_players']}")
        print(f"   • Hidden Gems Identified: {len(insights['hidden_gems_players'])}")
        print(f"   • Top All-Rounders: {len(insights['top_all_rounders'])}")
        
        # Recommendations
        print(f"\n🎯 RECRUITMENT RECOMMENDATIONS:")
        print(f"   1. 🏆 SPECIALIST RECRUITMENT:")
        print(f"      • Target elite performers in specific PSCDL areas")
        print(f"      • Immediate impact for tactical needs")
        
        if insights['hidden_gems_players']:
            print(f"   2. 💎 VALUE RECRUITMENT:")
            print(f"      • {len(insights['hidden_gems_players'])} undervalued players identified")
            print(f"      • High specialist potential at lower cost")
        
        print(f"   3. ⚖️ TACTICAL FLEXIBILITY:")
        print(f"      • {len(insights['top_all_rounders'])} all-rounders for multi-position roles")
        print(f"      • Injury cover and formation adaptability")
        
        print(f"\n🎉 RECRUITMENT ANALYSIS COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error in recruitment analysis: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Run statistical aggregation first: python3 03_statistical_aggregator.py")
        print("   2. Required statistics files in Statistics/ directory")
        print("   3. Proper data permissions")
        print("\n📝 Usage examples:")
        print("   from recruitment_analyzer import RecruitmentAnalyzer")
        print("   analyzer = RecruitmentAnalyzer(performance_df, totals_df)")
        print("   insights = analyzer.get_recruitment_insights()")


if __name__ == "__main__":
    main()
