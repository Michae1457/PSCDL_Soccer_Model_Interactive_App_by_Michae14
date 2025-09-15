"""
Statistical Aggregation Module

This module handles the aggregation of player-level match data into comprehensive
statistical datasets. It creates 5 different datasets measuring player performance
at different levels: totals, per-game, per-minute, per-90, and performance scores.

Author: Michael Xu
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StatisticalAggregator:
    """
    Handles aggregation of player-level match data into comprehensive statistical datasets
    """
    
    def __init__(self):
        """Initialize the statistical aggregator"""
        self.datasets = {}
        self.combined_data = None
    
    def load_all_player_data(self, player_folder_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load all player-level CSV files and combine them
        
        Args:
            player_folder_path (str): Path to folder containing player-level CSV files
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: (combined_data, match_files)
        """
        print("📊 Loading all player-level match data...")

        all_player_data = []
        match_files = []

        # Get all CSV files in the player-level folder
        player_files = [f for f in os.listdir(player_folder_path) if f.endswith('.csv')]
        print(f"Found {len(player_files)} player data files")

        for i, filename in enumerate(player_files):
            try:
                file_path = os.path.join(player_folder_path, filename)
                df = pd.read_csv(file_path)

                # Add match_id from filename
                match_id = filename.replace('match_', '').replace('.csv', '')
                df['match_id'] = match_id

                all_player_data.append(df)
                match_files.append(filename)

            except Exception as e:
                print(f"  ⚠️ Error loading {filename}: {e}")
                continue

        if not all_player_data:
            raise ValueError("No player data files could be loaded!")

        # Combine all dataframes
        print("🔄 Combining all player data...")
        combined_df = pd.concat(all_player_data, ignore_index=True)

        return combined_df, match_files

    def create_base_aggregation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create base aggregation with totals and basic info
        
        Args:
            df (pd.DataFrame): Combined player data from all matches
            
        Returns:
            pd.DataFrame: Base aggregated data with totals
        """
        print("🔄 Creating base aggregation...")

        # Define aggregation rules for totals and basic info
        aggregation_rules = {
            # Basic info - keep first occurrence
            'team': lambda x: list(x.unique()) if len(x.unique()) > 1 else x.iloc[0],

            # Count metrics - sum (map original names to total_ prefixed names)
            'minutes_played': 'sum',  # Will be renamed to total_minutes_played
            'total_passes': 'sum',
            'completed_passes': 'sum',  # Will be renamed to total_completed_passes
            'progressive_passes': 'sum',  # Will be renamed to total_progressive_passes
            'penalty_area_passes': 'sum',  # Will be renamed to total_penalty_area_passes
            'long_passes': 'sum',  # Will be renamed to total_long_passes
            'assists': 'sum',  # Will be renamed to total_assists
            'key_passes': 'sum',  # Will be renamed to total_key_passes
            'total_shots': 'sum',
            'shots_on_target': 'sum',  # Will be renamed to total_shots_on_target
            'goals': 'sum',  # Will be renamed to total_goals
            'total_xg': 'sum',
            'total_dribbles': 'sum',
            'successful_dribbles': 'sum',  # Will be renamed to total_successful_dribbles
            'total_duels': 'sum',
            'won_duels': 'sum',  # Will be renamed to total_won_duels
            'total_tackles': 'sum',
            'successful_tackles': 'sum',  # Will be renamed to total_successful_tackles
            'total_interceptions': 'sum',
            'total_ball_recoveries': 'sum',
            'total_clearances': 'sum',
            'total_blocks': 'sum',
            'fouls_committed': 'sum',  # Will be renamed to total_fouls_committed
            'fouls_won': 'sum',  # Will be renamed to total_fouls_won

            # Distance metrics - average
            'avg_pass_distance': 'mean',
            'avg_xg_per_shot': 'mean',

            # Match count
            'match_id': 'count'
        }

        # Group by player_id and player_name
        grouped = df.groupby(['player_id', 'player_name']).agg(aggregation_rules)

        # Rename match_id count to total_games
        grouped = grouped.rename(columns={'match_id': 'total_games'})
        grouped = grouped.reset_index()

        # Rename columns to have consistent total_ prefix
        column_rename_map = {
            'minutes_played': 'total_minutes_played',
            'completed_passes': 'total_completed_passes',
            'progressive_passes': 'total_progressive_passes',
            'penalty_area_passes': 'total_penalty_area_passes',
            'long_passes': 'total_long_passes',
            'assists': 'total_assists',
            'key_passes': 'total_key_passes',
            'shots_on_target': 'total_shots_on_target',
            'goals': 'total_goals',
            'successful_dribbles': 'total_successful_dribbles',
            'won_duels': 'total_won_duels',
            'successful_tackles': 'total_successful_tackles',
            'fouls_committed': 'total_fouls_committed',
            'fouls_won': 'total_fouls_won'
        }

        grouped = grouped.rename(columns=column_rename_map)

        # Handle team column
        def format_team_list(team_data):
            if isinstance(team_data, list):
                return ', '.join(team_data)
            else:
                return str(team_data)

        grouped['team'] = grouped['team'].apply(format_team_list)

        # Fill NaN values with 0
        numeric_columns = grouped.select_dtypes(include=[np.number]).columns
        grouped[numeric_columns] = grouped[numeric_columns].fillna(0)

        return grouped

    def create_per_game_dataset(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-game dataset with averages and rates
        
        Args:
            base_df (pd.DataFrame): Base aggregated data
            
        Returns:
            pd.DataFrame: Per-game statistics dataset
        """
        print("🔄 Creating per-game dataset...")

        # Start with basic identifiers
        per_game_df = base_df[['player_id', 'player_name', 'team', 'total_games']].copy()

        # Per-game averages (divide totals by total_games)
        count_metrics = [
            'total_minutes_played', 'total_passes', 'total_completed_passes',
            'total_progressive_passes', 'total_penalty_area_passes', 'total_long_passes',
            'total_assists', 'total_key_passes', 'total_shots', 'total_shots_on_target',
            'total_goals', 'total_xg', 'total_dribbles', 'total_successful_dribbles',
            'total_duels', 'total_won_duels', 'total_tackles', 'total_successful_tackles',
            'total_interceptions', 'total_ball_recoveries', 'total_clearances', 'total_blocks',
            'total_fouls_committed', 'total_fouls_won'
        ]

        for metric in count_metrics:
            if metric in base_df.columns:
                per_game_col = metric.replace('total_', '') + '_per_game'
                per_game_df[per_game_col] = (base_df[metric] / base_df['total_games'].replace(0, np.nan)).fillna(0)

        # Rate metrics (calculated from totals)
        per_game_df['pass_accuracy_per_game'] = (base_df['total_completed_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_game_df['dribble_success_rate_per_game'] = (base_df['total_successful_dribbles'] / base_df['total_dribbles'].replace(0, np.nan)).fillna(0)
        per_game_df['duel_win_rate_per_game'] = (base_df['total_won_duels'] / base_df['total_duels'].replace(0, np.nan)).fillna(0)
        per_game_df['tackle_success_rate_per_game'] = (base_df['total_successful_tackles'] / base_df['total_tackles'].replace(0, np.nan)).fillna(0)
        per_game_df['shot_accuracy_per_game'] = (base_df['total_shots_on_target'] / base_df['total_shots'].replace(0, np.nan)).fillna(0)
        per_game_df['progressive_pass_rate_per_game'] = (base_df['total_progressive_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_game_df['penalty_area_pass_rate_per_game'] = (base_df['total_penalty_area_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_game_df['long_pass_rate_per_game'] = (base_df['total_long_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)

        return per_game_df

    def create_per_minute_dataset(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-minute dataset
        
        Args:
            base_df (pd.DataFrame): Base aggregated data
            
        Returns:
            pd.DataFrame: Per-minute statistics dataset
        """
        print("🔄 Creating per-minute dataset...")

        # Start with basic identifiers
        per_minute_df = base_df[['player_id', 'player_name', 'team', 'total_minutes_played']].copy()

        # Per-minute averages (divide totals by total_minutes_played)
        count_metrics = [
            'total_passes', 'total_completed_passes', 'total_progressive_passes',
            'total_penalty_area_passes', 'total_long_passes', 'total_assists', 'total_key_passes',
            'total_shots', 'total_shots_on_target', 'total_goals', 'total_xg', 'total_dribbles',
            'total_successful_dribbles', 'total_duels', 'total_won_duels', 'total_tackles',
            'total_successful_tackles', 'total_interceptions', 'total_ball_recoveries',
            'total_clearances', 'total_blocks', 'total_fouls_committed', 'total_fouls_won'
        ]

        for metric in count_metrics:
            if metric in base_df.columns:
                per_minute_col = metric.replace('total_', '') + '_per_minute'
                per_minute_df[per_minute_col] = (base_df[metric] / base_df['total_minutes_played'].replace(0, np.nan)).fillna(0)

        # Rate metrics (same as per-game since they're ratios)
        per_minute_df['pass_accuracy_per_minute'] = (base_df['total_completed_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_minute_df['dribble_success_rate_per_minute'] = (base_df['total_successful_dribbles'] / base_df['total_dribbles'].replace(0, np.nan)).fillna(0)
        per_minute_df['duel_win_rate_per_minute'] = (base_df['total_won_duels'] / base_df['total_duels'].replace(0, np.nan)).fillna(0)
        per_minute_df['tackle_success_rate_per_minute'] = (base_df['total_successful_tackles'] / base_df['total_tackles'].replace(0, np.nan)).fillna(0)
        per_minute_df['shot_accuracy_per_minute'] = (base_df['total_shots_on_target'] / base_df['total_shots'].replace(0, np.nan)).fillna(0)
        per_minute_df['progressive_pass_rate_per_minute'] = (base_df['total_progressive_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_minute_df['penalty_area_pass_rate_per_minute'] = (base_df['total_penalty_area_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_minute_df['long_pass_rate_per_minute'] = (base_df['total_long_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)

        return per_minute_df

    def create_per_90_dataset(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-90 dataset
        
        Args:
            base_df (pd.DataFrame): Base aggregated data
            
        Returns:
            pd.DataFrame: Per-90 statistics dataset
        """
        print("🔄 Creating per-90 dataset...")

        # Start with basic identifiers
        per_90_df = base_df[['player_id', 'player_name', 'team', 'total_minutes_played']].copy()

        # Per-90 averages (divide totals by total_minutes_played and multiply by 90)
        count_metrics = [
            'total_passes', 'total_completed_passes', 'total_progressive_passes',
            'total_penalty_area_passes', 'total_long_passes', 'total_assists', 'total_key_passes',
            'total_shots', 'total_shots_on_target', 'total_goals', 'total_xg', 'total_dribbles',
            'total_successful_dribbles', 'total_duels', 'total_won_duels', 'total_tackles',
            'total_successful_tackles', 'total_interceptions', 'total_ball_recoveries',
            'total_clearances', 'total_blocks', 'total_fouls_committed', 'total_fouls_won'
        ]

        for metric in count_metrics:
            if metric in base_df.columns:
                per_90_col = metric.replace('total_', '') + '_per_90'
                per_90_df[per_90_col] = (base_df[metric] / base_df['total_minutes_played'].replace(0, np.nan) * 90).fillna(0)

        # Rate metrics (same as per-game since they're ratios)
        per_90_df['pass_accuracy_per_90'] = (base_df['total_completed_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_90_df['dribble_success_rate_per_90'] = (base_df['total_successful_dribbles'] / base_df['total_dribbles'].replace(0, np.nan)).fillna(0)
        per_90_df['duel_win_rate_per_90'] = (base_df['total_won_duels'] / base_df['total_duels'].replace(0, np.nan)).fillna(0)
        per_90_df['tackle_success_rate_per_90'] = (base_df['total_successful_tackles'] / base_df['total_tackles'].replace(0, np.nan)).fillna(0)
        per_90_df['shot_accuracy_per_90'] = (base_df['total_shots_on_target'] / base_df['total_shots'].replace(0, np.nan)).fillna(0)
        per_90_df['progressive_pass_rate_per_90'] = (base_df['total_progressive_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_90_df['penalty_area_pass_rate_per_90'] = (base_df['total_penalty_area_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)
        per_90_df['long_pass_rate_per_90'] = (base_df['total_long_passes'] / base_df['total_passes'].replace(0, np.nan)).fillna(0)

        return per_90_df

    def create_performance_scores_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance scores dataset with weighted averages
        
        Args:
            df (pd.DataFrame): Combined player data with performance scores
            
        Returns:
            pd.DataFrame: Performance scores dataset
        """
        print("🔄 Creating performance scores dataset...")

        # Define aggregation rules for performance scores
        score_aggregation_rules = {
            'team': lambda x: list(x.unique()) if len(x.unique()) > 1 else x.iloc[0],
            'performance_score': lambda x: np.average(x, weights=df.loc[x.index, 'minutes_played']) if 'minutes_played' in df.columns else x.mean(),
            'P_score': lambda x: np.average(x, weights=df.loc[x.index, 'minutes_played']) if 'minutes_played' in df.columns else x.mean(),
            'S_score': lambda x: np.average(x, weights=df.loc[x.index, 'minutes_played']) if 'minutes_played' in df.columns else x.mean(),
            'C_score': lambda x: np.average(x, weights=df.loc[x.index, 'minutes_played']) if 'minutes_played' in df.columns else x.mean(),
            'D_score': lambda x: np.average(x, weights=df.loc[x.index, 'minutes_played']) if 'minutes_played' in df.columns else x.mean(),
            'L_score': lambda x: np.average(x, weights=df.loc[x.index, 'minutes_played']) if 'minutes_played' in df.columns else x.mean(),
        }

        # Group by player_id and player_name
        grouped = df.groupby(['player_id', 'player_name']).agg(score_aggregation_rules)
        grouped = grouped.reset_index()

        # Handle team column
        def format_team_list(team_data):
            if isinstance(team_data, list):
                return ', '.join(team_data)
            else:
                return str(team_data)

        grouped['team'] = grouped['team'].apply(format_team_list)

        # Fill NaN values with 0
        numeric_columns = grouped.select_dtypes(include=[np.number]).columns
        grouped[numeric_columns] = grouped[numeric_columns].fillna(0)

        return grouped

    def aggregate_player_metrics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create all 5 datasets from the player data
        
        Args:
            df (pd.DataFrame): Combined player data from all matches
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary containing all 5 datasets
        """
        print("🔄 Creating structured player datasets...")

        # Create all datasets
        base_df = self.create_base_aggregation(df)
        per_game_df = self.create_per_game_dataset(base_df)
        per_minute_df = self.create_per_minute_dataset(base_df)
        per_90_df = self.create_per_90_dataset(base_df)
        performance_df = self.create_performance_scores_dataset(df)

        return {
            'totals': base_df,
            'per_game': per_game_df,
            'per_minute': per_minute_df,
            'per_90': per_90_df,
            'performance_scores': performance_df
        }

    def save_datasets(self, datasets: Dict[str, pd.DataFrame], output_dir: str) -> Dict[str, str]:
        """
        Save all datasets to CSV files
        
        Args:
            datasets (Dict[str, pd.DataFrame]): Dictionary of datasets to save
            output_dir (str): Output directory path
            
        Returns:
            Dict[str, str]: Dictionary mapping dataset names to file paths
        """
        print(f"\n💾 Saving structured datasets to {output_dir}/...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        output_files = {
            'totals': f'{output_dir}/statistics_totals.csv',
            'per_game': f'{output_dir}/statistics_per_game.csv',
            'per_minute': f'{output_dir}/statistics_per_minute.csv',
            'per_90': f'{output_dir}/statistics_per_90.csv',
            'performance_scores': f'{output_dir}/statistics_performance_scores.csv'
        }

        for dataset_name, df in datasets.items():
            output_file = output_files[dataset_name]
            df.to_csv(output_file, index=False)
            print(f"  ✅ {dataset_name}: {len(df)} players, {len(df.columns)} columns -> {output_file}")

        return output_files

    def aggregate_player_stats(self, player_folder_path: str, output_dir: str = 'Statistics') -> Dict[str, any]:
        """
        Main function to create structured player datasets
        
        Args:
            player_folder_path (str): Path to folder containing player-level CSV files
            output_dir (str): Output directory for statistics files
            
        Returns:
            Dict[str, any]: Aggregation results and statistics
        """
        print("🚀 Creating Structured Player-Level Datasets")
        print("=" * 60)

        try:
            # Check if player folder exists
            if not os.path.exists(player_folder_path):
                raise ValueError(f"Player folder not found: {player_folder_path}")

            # Step 1: Load all player data
            combined_df, match_files = self.load_all_player_data(player_folder_path)
            self.combined_data = combined_df

            # Step 2: Create structured datasets
            datasets = self.aggregate_player_metrics(combined_df)
            self.datasets = datasets

            # Step 3: Save all datasets
            output_files = self.save_datasets(datasets, output_dir)

            # Return results
            results = {
                "success": True,
                "total_players": len(combined_df),
                "unique_players": len(datasets['totals']),
                "match_files_processed": len(match_files),
                "datasets_created": len(datasets),
                "output_files": output_files,
                "datasets": datasets
            }

            print(f"\n🎉 Aggregation completed successfully!")
            print(f"📊 Processed {len(match_files)} match files")
            print(f"👥 Found {results['unique_players']} unique players")
            print(f"📈 Created {results['datasets_created']} statistical datasets")

            return results

        except Exception as e:
            print(f"❌ Error creating structured datasets: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}


def main():
    """
    Main function to run statistical aggregation
    """
    print("⚽ Soccer Analytics Pipeline - Statistical Aggregation Module")
    print("=" * 70)
    
    try:
        # Initialize aggregator
        aggregator = StatisticalAggregator()
        
        # Define paths (adjust these paths as needed)
        player_folder_path = 'Soccer Analytics Pipeline/Match/Player-Level'
        output_dir = 'Soccer Analytics Pipeline/Statistics'
        
        # Run aggregation
        results = aggregator.aggregate_player_stats(player_folder_path, output_dir)
        
        if results["success"]:
            print(f"\n🎉 Statistical aggregation completed successfully!")
            print(f"📊 Created datasets for {results['unique_players']} players")
        else:
            print(f"\n❌ Statistical aggregation failed: {results['error']}")
            
    except Exception as e:
        print(f"❌ Fatal error in statistical aggregation: {e}")


if __name__ == "__main__":
    main()
