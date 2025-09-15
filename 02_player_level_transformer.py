"""
Player Level Data Transformation Module

This module handles the conversion from match-level events data to player-level statistics.
It includes the PlayerLevelAnalyzer class and advanced performance scoring system (PSCDL framework).

Author: Michael Xu
"""

import pandas as pd
import numpy as np
import ast
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import os
import importlib.util
import sys
from pathlib import Path

# Import the data preparation module properly
def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load the data preparation module
data_prep = load_module("data_preparation", "Soccer Analytics Pipeline/01_data_preparation.py")
StatsBombDataScraper = data_prep.StatsBombDataScraper


class PlayerLevelAnalyzer:
    """
    Convert match-level events data to player-level statistics for soccer analytics
    """

    def __init__(self, events_df: pd.DataFrame):
        """
        Initialize the analyzer with events data
        
        Args:
            events_df (pd.DataFrame): Match-level events data
        """
        self.events_df = events_df.copy()
        self.player_stats = None

    def parse_location(self, location_data) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse location data to get x, y coordinates
        
        Args:
            location_data: Location data in various formats
            
        Returns:
            Tuple[Optional[float], Optional[float]]: (x, y) coordinates
        """
        try:
            # Handle different data types
            if location_data is None or (isinstance(location_data, float) and np.isnan(location_data)):
                return None, None
            elif isinstance(location_data, str):
                location = ast.literal_eval(location_data)
                return location[0], location[1]
            elif isinstance(location_data, (list, tuple, np.ndarray)):
                return location_data[0], location_data[1]
            else:
                return None, None
        except:
            return None, None

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> Optional[float]:
        """
        Calculate Euclidean distance between two points
        
        Args:
            x1, y1: Starting coordinates
            x2, y2: Ending coordinates
            
        Returns:
            Optional[float]: Distance between points
        """
        if any(val is None or (isinstance(val, float) and np.isnan(val)) for val in [x1, y1, x2, y2]):
            return None
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def is_progressive_pass(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
        """
        Determine if a pass is progressive (moves ball significantly forward)
        
        Args:
            start_x, start_y: Starting coordinates
            end_x, end_y: Ending coordinates
            
        Returns:
            bool: True if pass is progressive
        """
        if any(val is None or (isinstance(val, float) and np.isnan(val)) for val in [start_x, start_y, end_x, end_y]):
            return False
        # Progressive if ball moves at least 10 yards forward
        return (end_x - start_x) >= 10

    def is_penalty_area_pass(self, end_x: float, end_y: float) -> bool:
        """
        Check if pass ends in penalty area
        
        Args:
            end_x, end_y: Ending coordinates
            
        Returns:
            bool: True if pass ends in penalty area
        """
        if (end_x is None or (isinstance(end_x, float) and np.isnan(end_x)) or
            end_y is None or (isinstance(end_y, float) and np.isnan(end_y))):
            return False
        # Penalty area roughly from x=100 to x=120, y=18 to y=62
        return 100 <= end_x <= 120 and 18 <= end_y <= 62

    def aggregate_player_stats(self) -> pd.DataFrame:
        """
        Aggregate match events to player-level statistics
        
        Returns:
            pd.DataFrame: Processed player events with calculated metrics
        """
        print("Starting player-level aggregation...")

        # Filter out non-player events
        player_events = self.events_df[
            (self.events_df['player'].notna()) &
            (self.events_df['player'] != '')
        ].copy()

        # Parse locations for all events
        player_events['start_x'] = player_events['location'].apply(
            lambda x: self.parse_location(x)[0]
        )
        player_events['start_y'] = player_events['location'].apply(
            lambda x: self.parse_location(x)[1]
        )

        # Parse pass end locations
        player_events['end_x'] = player_events['pass_end_location'].apply(
            lambda x: self.parse_location(x)[0]
        )
        player_events['end_y'] = player_events['pass_end_location'].apply(
            lambda x: self.parse_location(x)[1]
        )

        # Calculate pass distances
        player_events['pass_distance'] = player_events.apply(
            lambda row: self.calculate_distance(
                row['start_x'], row['start_y'],
                row['end_x'], row['end_y']
            ), axis=1
        )

        # Calculate progressive passes
        player_events['is_progressive_pass'] = player_events.apply(
            lambda row: self.is_progressive_pass(
                row['start_x'], row['start_y'],
                row['end_x'], row['end_y']
            ), axis=1
        )

        # Calculate penalty area passes
        player_events['is_penalty_area_pass'] = player_events.apply(
            lambda row: self.is_penalty_area_pass(row['end_x'], row['end_y']), axis=1
        )

        print(f"✅ Processed {len(player_events)} player events")
        return player_events

    def calculate_player_statistics(self, player_events: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive player statistics from processed events
        
        Args:
            player_events (pd.DataFrame): Processed player events
            
        Returns:
            pd.DataFrame: Player-level statistics
        """
        print("Calculating player statistics...")

        # Group by player and aggregate statistics
        player_stats = []

        for player_id in player_events['player_id'].unique():
            if pd.isna(player_id):
                continue

            player_data = player_events[player_events['player_id'] == player_id]
            player_name = player_data['player'].iloc[0]
            team = player_data['team'].iloc[0]

            # Basic counts
            total_events = len(player_data)
            minutes_played = player_data['minute'].max() - player_data['minute'].min() + 1

            # Pass statistics
            passes = player_data[player_data['type'] == 'Pass']
            total_passes = len(passes)
            if 'pass_outcome' in passes.columns:
                completed_passes = len(passes[passes['pass_outcome'].isna()])  # No outcome = completed
            else:
                completed_passes = total_passes  # Assume all passes completed if no outcome column
            pass_accuracy = completed_passes / total_passes if total_passes > 0 else 0

            # Progressive passes
            progressive_passes = len(passes[passes['is_progressive_pass'] == True])

            # Penalty area passes
            penalty_area_passes = len(passes[passes['is_penalty_area_pass'] == True])

            # Pass distance statistics
            if 'pass_distance' in passes.columns:
                pass_distances = passes['pass_distance'].dropna()
                avg_pass_distance = pass_distances.mean() if len(pass_distances) > 0 else 0
                long_passes = len(passes[passes['pass_distance'] > 30])
            else:
                avg_pass_distance = 0
                long_passes = 0

            # Shot statistics
            shots = player_data[player_data['type'] == 'Shot']
            total_shots = len(shots)
            if 'shot_outcome' in shots.columns:
                shots_on_target = len(shots[shots['shot_outcome'].isin(['Goal', 'Saved'])])
                goals = len(shots[shots['shot_outcome'] == 'Goal'])
            else:
                shots_on_target = 0
                goals = 0

            # xG statistics
            if 'shot_statsbomb_xg' in shots.columns:
                xg_values = shots['shot_statsbomb_xg'].dropna()
                total_xg = xg_values.sum() if len(xg_values) > 0 else 0
                avg_xg_per_shot = xg_values.mean() if len(xg_values) > 0 else 0
            else:
                total_xg = 0
                avg_xg_per_shot = 0

            # Dribble statistics
            dribbles = player_data[player_data['type'] == 'Dribble']
            total_dribbles = len(dribbles)
            if 'dribble_outcome' in dribbles.columns:
                successful_dribbles = len(dribbles[dribbles['dribble_outcome'] == 'Complete'])
            else:
                successful_dribbles = 0
            dribble_success_rate = successful_dribbles / total_dribbles if total_dribbles > 0 else 0

            # Defensive statistics
            duels = player_data[player_data['type'] == 'Duel']
            total_duels = len(duels)
            if 'duel_outcome' in duels.columns:
                won_duels = len(duels[duels['duel_outcome'] == 'Won'])
            else:
                won_duels = 0
            duel_win_rate = won_duels / total_duels if total_duels > 0 else 0

            # Tackles (duels with type 'Tackle')
            if 'duel_type' in duels.columns:
                tackles = duels[duels['duel_type'] == 'Tackle']
                total_tackles = len(tackles)
                if 'duel_outcome' in tackles.columns:
                    successful_tackles = len(tackles[tackles['duel_outcome'] == 'Won'])
                else:
                    successful_tackles = 0
            else:
                tackles = pd.DataFrame()
                total_tackles = 0
                successful_tackles = 0
            tackle_success_rate = successful_tackles / total_tackles if total_tackles > 0 else 0

            # Interceptions
            interceptions = player_data[player_data['type'] == 'Interception']
            total_interceptions = len(interceptions)

            # Ball recoveries
            ball_recoveries = player_data[player_data['type'] == 'Ball Recovery']
            total_ball_recoveries = len(ball_recoveries)

            # Clearances
            clearances = player_data[player_data['type'] == 'Clearance']
            total_clearances = len(clearances)

            # Blocks
            blocks = player_data[player_data['type'] == 'Block']
            total_blocks = len(blocks)

            # Fouls
            fouls_committed = len(player_data[player_data['type'] == 'Foul Committed'])
            fouls_won = len(player_data[player_data['type'] == 'Foul Won'])

            # Assists (passes that led to goals)
            assists = len(passes[passes['pass_goal_assist'] == True]) if 'pass_goal_assist' in passes.columns else 0

            # Key passes (passes that led to shots)
            key_passes = len(passes[passes['pass_shot_assist'] == True]) if 'pass_shot_assist' in passes.columns else 0

            # Create player statistics dictionary
            player_stat = {
                'player_id': player_id,
                'player_name': player_name,
                'team': team,
                'minutes_played': minutes_played,
                'total_events': total_events,

                # Passing metrics
                'total_passes': total_passes,
                'completed_passes': completed_passes,
                'pass_accuracy': pass_accuracy,
                'progressive_passes': progressive_passes,
                'penalty_area_passes': penalty_area_passes,
                'avg_pass_distance': avg_pass_distance,
                'long_passes': long_passes,
                'assists': assists,
                'key_passes': key_passes,

                # Shooting metrics
                'total_shots': total_shots,
                'shots_on_target': shots_on_target,
                'goals': goals,
                'total_xg': total_xg,
                'avg_xg_per_shot': avg_xg_per_shot,
                'shot_accuracy': shots_on_target / total_shots if total_shots > 0 else 0,

                # Dribbling metrics
                'total_dribbles': total_dribbles,
                'successful_dribbles': successful_dribbles,
                'dribble_success_rate': dribble_success_rate,

                # Defensive metrics
                'total_duels': total_duels,
                'won_duels': won_duels,
                'duel_win_rate': duel_win_rate,
                'total_tackles': total_tackles,
                'successful_tackles': successful_tackles,
                'tackle_success_rate': tackle_success_rate,
                'total_interceptions': total_interceptions,
                'total_ball_recoveries': total_ball_recoveries,
                'total_clearances': total_clearances,
                'total_blocks': total_blocks,

                # Discipline
                'fouls_committed': fouls_committed,
                'fouls_won': fouls_won,
            }

            player_stats.append(player_stat)

        self.player_stats = pd.DataFrame(player_stats)
        print(f"✅ Created player-level statistics for {len(self.player_stats)} players")
        return self.player_stats

    def _calculate_passing_creation_score(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Calculate Passing/Creation (P) subscore
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            
        Returns:
            Tuple[pd.Series, Dict]: (P_score, percentile_dict)
        """
        # Create combined key pass/assist metric (per 90)
        if 'key_passes' in df.columns and 'assists' in df.columns and 'minutes_played' in df.columns:
            df['key_passes_assists_p90'] = ((df['key_passes'] + df['assists']) / df['minutes_played'] * 90).fillna(0)
            key_pass_assist_p90 = df['key_passes_assists_p90'].rank(pct=True) * 100
        else:
            key_pass_assist_p90 = pd.Series([50] * len(df), index=df.index)

        # Calculate progressive passes per 90 percentile
        if 'progressive_passes' in df.columns and 'minutes_played' in df.columns:
            df['progressive_passes_p90'] = (df['progressive_passes'] / df['minutes_played'] * 90).fillna(0)
            prog_pass_p90 = df['progressive_passes_p90'].rank(pct=True) * 100
        else:
            prog_pass_p90 = pd.Series([50] * len(df), index=df.index)

        # Pass accuracy percentile
        if 'pass_accuracy' in df.columns:
            pass_acc_p = df['pass_accuracy'].rank(pct=True) * 100
        else:
            pass_acc_p = pd.Series([50] * len(df), index=df.index)

        # Calculate P subscore
        P_score = (
            pass_acc_p * 0.35 +
            prog_pass_p90 * 0.35 +
            key_pass_assist_p90 * 0.30
        )

        return P_score, {
            'pass_accuracy_p': pass_acc_p,
            'progressive_passes_p90_p': prog_pass_p90,
            'key_passes_assists_p90_p': key_pass_assist_p90
        }

    def _calculate_finishing_shot_score(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Calculate Finishing/Shot Value (S) subscore
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            
        Returns:
            Tuple[pd.Series, Dict]: (S_score, percentile_dict)
        """
        # Calculate xG per 90
        if 'total_xg' in df.columns and 'minutes_played' in df.columns:
            df['total_xg_p90'] = (df['total_xg'] / df['minutes_played'] * 90).fillna(0)
            xg_p90 = df['total_xg_p90'].rank(pct=True) * 100
        else:
            xg_p90 = pd.Series([50] * len(df), index=df.index)

        # Shot on target rate
        if 'shot_accuracy' in df.columns:
            sot_rate = df['shot_accuracy'].rank(pct=True) * 100
        else:
            sot_rate = pd.Series([50] * len(df), index=df.index)

        # Goals minus xG per 90 (finishing ability)
        if 'goals' in df.columns and 'total_xg' in df.columns and 'minutes_played' in df.columns:
            df['goals_p90'] = (df['goals'] / df['minutes_played'] * 90).fillna(0)
            df['goals_minus_xg_p90'] = df['goals_p90'] - df['total_xg_p90']
            goals_minus_xg_p90 = df['goals_minus_xg_p90'].rank(pct=True) * 100
        else:
            goals_minus_xg_p90 = pd.Series([50] * len(df), index=df.index)

        # Calculate S subscore
        S_score = (
            xg_p90 * 0.50 +
            sot_rate * 0.30 +
            goals_minus_xg_p90 * 0.20
        )

        return S_score, {
            'total_xg_p90_p': xg_p90,
            'shots_on_target_rate_p': sot_rate,
            'goals_minus_xg_p90_p': goals_minus_xg_p90
        }

    def _calculate_carrying_1v1_score(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Calculate Carrying/1v1 (C) subscore
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            
        Returns:
            Tuple[pd.Series, Dict]: (C_score, percentile_dict)
        """
        # Successful dribbles per 90
        if 'successful_dribbles' in df.columns and 'minutes_played' in df.columns:
            df['successful_dribbles_p90'] = (df['successful_dribbles'] / df['minutes_played'] * 90).fillna(0)
            succ_dribbles_p90 = df['successful_dribbles_p90'].rank(pct=True) * 100
        else:
            succ_dribbles_p90 = pd.Series([50] * len(df), index=df.index)

        # Dribble success rate
        if 'dribble_success_rate' in df.columns:
            dribble_rate = df['dribble_success_rate'].rank(pct=True) * 100
        else:
            dribble_rate = pd.Series([50] * len(df), index=df.index)

        # Calculate C subscore
        C_score = (
            succ_dribbles_p90 * 0.60 +
            dribble_rate * 0.40
        )

        return C_score, {
            'successful_dribbles_p90_p': succ_dribbles_p90,
            'dribble_success_rate_p': dribble_rate
        }

    def _calculate_defending_disruption_score(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Calculate Defending/Disruption (D) subscore
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            
        Returns:
            Tuple[pd.Series, Dict]: (D_score, percentile_dict)
        """
        # Successful tackles per 90
        if 'successful_tackles' in df.columns and 'minutes_played' in df.columns:
            df['successful_tackles_p90'] = (df['successful_tackles'] / df['minutes_played'] * 90).fillna(0)
            tackles_succ_p90 = df['successful_tackles_p90'].rank(pct=True) * 100
        else:
            tackles_succ_p90 = pd.Series([50] * len(df), index=df.index)

        # Tackle success rate
        if 'tackle_success_rate' in df.columns:
            tackle_rate = df['tackle_success_rate'].rank(pct=True) * 100
        else:
            tackle_rate = pd.Series([50] * len(df), index=df.index)

        # Interceptions per 90
        if 'total_interceptions' in df.columns and 'minutes_played' in df.columns:
            df['interceptions_p90'] = (df['total_interceptions'] / df['minutes_played'] * 90).fillna(0)
            interceptions_p90 = df['interceptions_p90'].rank(pct=True) * 100
        else:
            interceptions_p90 = pd.Series([50] * len(df), index=df.index)

        # Ball recoveries per 90
        if 'total_ball_recoveries' in df.columns and 'minutes_played' in df.columns:
            df['recoveries_p90'] = (df['total_ball_recoveries'] / df['minutes_played'] * 90).fillna(0)
            recoveries_p90 = df['recoveries_p90'].rank(pct=True) * 100
        else:
            recoveries_p90 = pd.Series([50] * len(df), index=df.index)

        # Blocks per 90
        if 'total_blocks' in df.columns and 'minutes_played' in df.columns:
            df['blocks_p90'] = (df['total_blocks'] / df['minutes_played'] * 90).fillna(0)
            blocks_p90 = df['blocks_p90'].rank(pct=True) * 100
        else:
            blocks_p90 = pd.Series([50] * len(df), index=df.index)

        # Calculate D subscore
        D_score = (
            tackles_succ_p90 * 0.30 +
            tackle_rate * 0.25 +
            interceptions_p90 * 0.20 +
            recoveries_p90 * 0.15 +
            blocks_p90 * 0.10
        )

        return D_score, {
            'successful_tackles_p90_p': tackles_succ_p90,
            'tackle_success_rate_p': tackle_rate,
            'interceptions_p90_p': interceptions_p90,
            'recoveries_p90_p': recoveries_p90,
            'blocks_p90_p': blocks_p90
        }

    def _calculate_discipline_score(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """
        Calculate Discipline (L) subscore as malus/bonus
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            
        Returns:
            Tuple[pd.Series, Dict]: (L_score, percentile_dict)
        """
        # Fouls per 90 (lower is better)
        if 'fouls_committed' in df.columns and 'minutes_played' in df.columns:
            df['fouls_p90'] = (df['fouls_committed'] / df['minutes_played'] * 90).fillna(0)
            # For discipline, lower fouls = higher score
            fouls_p90 = (1 - df['fouls_p90'].rank(pct=True)) * 100
        else:
            fouls_p90 = pd.Series([50] * len(df), index=df.index)

        # Calculate L subscore (discipline bonus/malus)
        L_score = fouls_p90  # Already inverted (100 - fouls percentile)

        return L_score, {
            'fouls_p90_p': fouls_p90
        }

    def _calculate_advanced_performance_scores(self, df: pd.DataFrame, coefficients: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate the advanced performance score using 5 subscores (PSCDL framework)
        
        Args:
            df (pd.DataFrame): Player statistics dataframe
            coefficients (Optional[Dict]): Custom coefficients for subscore weighting
            
        Returns:
            pd.DataFrame: DataFrame with performance scores added
        """
        if coefficients is None:
            # Default balanced coefficients
            coefficients = {
                'P': 0.25,  # Passing/Creation
                'S': 0.25,  # Finishing/Shot Value
                'C': 0.20,  # Carrying/1v1
                'D': 0.20,  # Defending/Disruption
                'L': 0.10   # Discipline (bonus/malus)
            }

        # Calculate each subscore
        P_score, _ = self._calculate_passing_creation_score(df)
        S_score, _ = self._calculate_finishing_shot_score(df)
        C_score, _ = self._calculate_carrying_1v1_score(df)
        D_score, _ = self._calculate_defending_disruption_score(df)
        L_score, _ = self._calculate_discipline_score(df)

        # Calculate main performance score
        performance_score = (
            P_score * coefficients['P'] +
            S_score * coefficients['S'] +
            C_score * coefficients['C'] +
            D_score * coefficients['D'] +
            L_score * coefficients['L']
        )

        # Add all scores to dataframe
        df['P_score'] = P_score
        df['S_score'] = S_score
        df['C_score'] = C_score
        df['D_score'] = D_score
        df['L_score'] = L_score
        df['performance_score'] = performance_score

        return df

    def create_advanced_metrics(self, coefficients: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create advanced metrics and features for modeling using the 5-subscore performance system
        
        Args:
            coefficients (Optional[Dict]): Custom coefficients for subscore weighting
            
        Returns:
            pd.DataFrame: DataFrame with advanced metrics and performance scores
        """
        if self.player_stats is None:
            raise ValueError("Must run aggregate_player_stats() first")

        print("Creating advanced metrics with 5-subscore performance system...")

        df = self.player_stats.copy()
        
        # Ensure all numeric columns are properly converted to numeric types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # Per 90 minutes metrics - ensure numeric conversion first
        numeric_columns = ['total_passes', 'total_shots', 'total_dribbles', 'total_tackles', 
                          'total_interceptions', 'total_xg', 'minutes_played']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Now calculate per-90 metrics safely
        df['passes_per_90'] = (df['total_passes'] / df['minutes_played'].replace(0, np.nan)) * 90
        df['shots_per_90'] = (df['total_shots'] / df['minutes_played'].replace(0, np.nan)) * 90
        df['dribbles_per_90'] = (df['total_dribbles'] / df['minutes_played'].replace(0, np.nan)) * 90
        df['tackles_per_90'] = (df['total_tackles'] / df['minutes_played'].replace(0, np.nan)) * 90
        df['interceptions_per_90'] = (df['total_interceptions'] / df['minutes_played'].replace(0, np.nan)) * 90
        df['xg_per_90'] = (df['total_xg'] / df['minutes_played'].replace(0, np.nan)) * 90
        
        # Fill NaN values with 0
        per_90_columns = ['passes_per_90', 'shots_per_90', 'dribbles_per_90', 'tackles_per_90', 'interceptions_per_90', 'xg_per_90']
        for col in per_90_columns:
            df[col] = df[col].fillna(0)

        # Efficiency metrics - ensure numeric conversion
        efficiency_columns = ['progressive_passes', 'penalty_area_passes', 'long_passes']
        for col in efficiency_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['progressive_pass_rate'] = df['progressive_passes'] / df['total_passes'].replace(0, np.nan)
        df['penalty_area_pass_rate'] = df['penalty_area_passes'] / df['total_passes'].replace(0, np.nan)
        df['long_pass_rate'] = df['long_passes'] / df['total_passes'].replace(0, np.nan)
        
        # Fill NaN values with 0 for rate metrics
        rate_columns = ['progressive_pass_rate', 'penalty_area_pass_rate', 'long_pass_rate']
        for col in rate_columns:
            df[col] = df[col].fillna(0)

        # Defensive efficiency - ensure numeric conversion
        defensive_columns = ['total_clearances', 'total_blocks', 'key_passes']
        for col in defensive_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['defensive_actions_per_90'] = (
            (df['total_tackles'] + df['total_interceptions'] + df['total_clearances'] + df['total_blocks'])
            / df['minutes_played'].replace(0, np.nan) * 90
        ).fillna(0)

        # Attacking efficiency
        df['attacking_actions_per_90'] = (
            (df['total_shots'] + df['total_dribbles'] + df['key_passes'])
            / df['minutes_played'].replace(0, np.nan) * 90
        ).fillna(0)

        # Calculate advanced performance scores using 5-subscore system
        df = self._calculate_advanced_performance_scores(df, coefficients)

        self.player_stats = df
        return self.player_stats

    def clean_data(self) -> pd.DataFrame:
        """
        Clean and validate the player-level data
        
        Returns:
            pd.DataFrame: Cleaned player statistics
        """
        if self.player_stats is None:
            raise ValueError("Must run aggregate_player_stats() first")

        df = self.player_stats.copy()

        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        # Handle infinite values
        df = df.replace([np.inf, -np.inf], 0)

        # Remove players with very few minutes (less than 10 minutes)
        df = df[df['minutes_played'] >= 10]

        # Cap extreme values (outliers)
        for col in ['passes_per_90', 'shots_per_90', 'dribbles_per_90', 'tackles_per_90']:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=q99)

        # Ensure rates are between 0 and 1
        rate_columns = ['pass_accuracy', 'dribble_success_rate', 'duel_win_rate',
                       'tackle_success_rate', 'shot_accuracy', 'progressive_pass_rate',
                       'penalty_area_pass_rate', 'long_pass_rate']

        for col in rate_columns:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)

        self.player_stats = df
        return self.player_stats


def transform_to_player_level(events_df: pd.DataFrame, coefficients: Optional[Dict] = None) -> Tuple[PlayerLevelAnalyzer, pd.DataFrame]:
    """
    Main function to run the player-level analysis
    
    Args:
        events_df (pd.DataFrame): Match-level events data
        coefficients (Optional[Dict]): Custom coefficients for performance scoring
        
    Returns:
        Tuple[PlayerLevelAnalyzer, pd.DataFrame]: (analyzer, player_stats)
    """
    analyzer = PlayerLevelAnalyzer(events_df)

    # Step 1: Process events and aggregate player statistics
    player_events = analyzer.aggregate_player_stats()
    player_stats = analyzer.calculate_player_statistics(player_events)

    # Step 2: Create advanced metrics with performance scoring
    player_stats = analyzer.create_advanced_metrics(coefficients)

    # Step 3: Clean data
    player_stats = analyzer.clean_data()

    return analyzer, player_stats


def main():
    """
    Main function to demonstrate player level transformation
    """
    print("⚽ Soccer Analytics Pipeline - Player Level Transformation Module")
    print("=" * 70)

    try:
        # Initialize the data scraper to get match IDs
        scraper = StatsBombDataScraper()
        matches_ids = scraper.get_all_match_ids()
        
        if not matches_ids:
            print("❌ No match IDs found. Please check your StatsBomb connection.")
            return

        EVENT_DIR = "Soccer Analytics Pipeline/Match/Event-Level"
        PLAYER_DIR = "Soccer Analytics Pipeline/Match/Player-Level"

        os.makedirs(EVENT_DIR, exist_ok=True)
        os.makedirs(PLAYER_DIR, exist_ok=True)

        print(f"Processing {len(matches_ids)} matches...")
        print("=" * 50)

        successful_transformations = 0
        failed_transformations = 0

        for i, match_id in enumerate(matches_ids):
            print(f"Processing match {i+1}/{len(matches_ids)}: {match_id}")

            try:
                # Fetch event data using the scraper
                events = scraper.fetch_match_events(match_id)
                
                if events.empty:
                    print(f"  ⚠️ No events data for match {match_id}")
                    failed_transformations += 1
                    continue
                
                # Save event data (convert string to Path object)
                success, message = scraper.save_match_data(match_id, events, Path(EVENT_DIR))
                
                if not success:
                    print(f"  ❌ {message}")
                    failed_transformations += 1
                    continue
                
                print(f"  ✅ {message}")

                # Transform to player level
                analyzer, player_stats = transform_to_player_level(events)
                
                # Save player statistics
                player_file = f'{PLAYER_DIR}/match_{match_id}.csv'
                player_stats.to_csv(player_file, index=False)
                print(f"  ✅ Player stats saved: {len(player_stats)} players")
                
                successful_transformations += 1
                print("-" * 50)

            except Exception as e:
                print(f"  ❌ Error processing match {match_id}: {e}")
                failed_transformations += 1
                continue
        
        print(f"\n📊 TRANSFORMATION COMPLETE!")
        print("=" * 50)
        print(f"✅ Successful transformations: {successful_transformations}")
        print(f"❌ Failed transformations: {failed_transformations}")
        print(f"📁 Event data saved to: {EVENT_DIR}")
        print(f"📁 Player data saved to: {PLAYER_DIR}")

    except Exception as e:
        print(f"❌ Error in main function: {e}")
        print("\n💡 Make sure you have:")
        print("   1. statsbombpy installed: pip install statsbombpy")
        print("   2. Internet connection for StatsBomb API")
        print("   3. Proper file permissions for creating directories")


if __name__ == "__main__":
    main()
