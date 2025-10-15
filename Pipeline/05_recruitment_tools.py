"""
Visualization and Dashboard Tools Module

This module provides comprehensive visualization tools for soccer analytics,
including player comparison dashboards, performance analysis charts, and
interactive scouting tools.

Author: Michael Xu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")


class SoccerRecruitmentTools:
    """
    Comprehensive visualization tools for soccer analytics
    """
    
    def __init__(self):
        """Initialize visualization tools"""
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                      '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
    def plot_performance_distribution(self, recruitment_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot performance score distribution
        
        Args:
            recruitment_df (pd.DataFrame): Recruitment data with performance scores
            save_path (Optional[str]): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create histogram
        ax.hist(recruitment_df['performance_score'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add statistical lines
        mean_score = recruitment_df['performance_score'].mean()
        top_10_threshold = recruitment_df['performance_score'].quantile(0.9)
        
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_score:.1f}')
        ax.axvline(top_10_threshold, color='orange', linestyle='--', linewidth=2,
                  label=f'Top 10%: {top_10_threshold:.1f}')
        
        ax.set_title('Overall Performance Score Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Performance Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Total Players: {len(recruitment_df):,}\\n'
        stats_text += f'Mean: {mean_score:.2f}\\n'
        stats_text += f'Median: {recruitment_df["performance_score"].median():.2f}\\n'
        stats_text += f'Std: {recruitment_df["performance_score"].std():.2f}'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Performance distribution plot saved to: {save_path}")
        
        plt.show()
    
    def plot_top_performers(self, totals_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot top performers in key categories
        
        Args:
            totals_df (pd.DataFrame): Totals statistics dataset
            save_path (Optional[str]): Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Sort and get top 10 for each category
        top_scorers = totals_df.sort_values(by='total_goals', ascending=False).head(10)
        top_assisters = totals_df.sort_values(by='total_assists', ascending=False).head(10)
        top_interceptors = totals_df.sort_values(by='total_interceptions', ascending=False).head(10)
        
        # Plot Top Scorers
        sns.barplot(x='total_goals', y='player_name', data=top_scorers, color='blue', ax=axes[0])
        axes[0].set_title('Top 10 Scorers (Total Goals)', fontweight='bold')
        axes[0].set_xlabel('Total Goals')
        axes[0].set_ylabel('Player Name')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.0f')
        
        # Plot Top Assisters
        sns.barplot(x='total_assists', y='player_name', data=top_assisters, color='red', ax=axes[1])
        axes[1].set_title('Top 10 Assisters (Total Assists)', fontweight='bold')
        axes[1].set_xlabel('Total Assists')
        axes[1].set_ylabel('')
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.0f')
        
        # Plot Top Interceptors
        sns.barplot(x='total_interceptions', y='player_name', data=top_interceptors, color='green', ax=axes[2])
        axes[2].set_title('Top 10 Interceptors (Total Interceptions)', fontweight='bold')
        axes[2].set_xlabel('Total Interceptions')
        axes[2].set_ylabel('')
        for container in axes[2].containers:
            axes[2].bar_label(container, fmt='%.0f')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Top performers plot saved to: {save_path}")
        
        plt.show()
    
    def plot_per_90_comparison(self, per_90_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot per-90 minute comparison charts
        
        Args:
            per_90_df (pd.DataFrame): Per-90 statistics dataset
            save_path (Optional[str]): Path to save the plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Sort and get top 10 for each category
        top_scorers_per_90 = per_90_df.sort_values(by='goals_per_90', ascending=False).head(10)
        top_assisters_per_90 = per_90_df.sort_values(by='assists_per_90', ascending=False).head(10)
        top_interceptors_per_90 = per_90_df.sort_values(by='interceptions_per_90', ascending=False).head(10)
        
        # Plot Top Scorers Per 90
        sns.barplot(x='goals_per_90', y='player_name', data=top_scorers_per_90, color='blue', ax=axes[0])
        axes[0].set_title('Top 10 Scorers Per 90', fontweight='bold')
        axes[0].set_xlabel('Goals Per 90')
        axes[0].set_ylabel('Player Name')
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%.2f')
        
        # Plot Top Assisters Per 90
        sns.barplot(x='assists_per_90', y='player_name', data=top_assisters_per_90, color='red', ax=axes[1])
        axes[1].set_title('Top 10 Assisters Per 90', fontweight='bold')
        axes[1].set_xlabel('Assists Per 90')
        axes[1].set_ylabel('')
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt='%.2f')
        
        # Plot Top Interceptors Per 90
        sns.barplot(x='interceptions_per_90', y='player_name', data=top_interceptors_per_90, color='green', ax=axes[2])
        axes[2].set_title('Top 10 Interceptors Per 90', fontweight='bold')
        axes[2].set_xlabel('Interceptions Per 90')
        axes[2].set_ylabel('')
        for container in axes[2].containers:
            axes[2].bar_label(container, fmt='%.2f')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Per-90 comparison plot saved to: {save_path}")
        
        plt.show()
    
    def create_radar_comparison(self, performance_df: pd.DataFrame, player_names: List[str], 
                              save_path: Optional[str] = None):
        """
        Create radar chart comparison for specific players
        
        Args:
            performance_df (pd.DataFrame): Performance scores dataset
            player_names (List[str]): List of player names to compare
            save_path (Optional[str]): Path to save the plot
        """
        # Filter data for selected players
        comparison_data = performance_df[performance_df['player_name'].isin(player_names)]
        
        if len(comparison_data) == 0:
            print("❌ No players found in the dataset")
            return
        
        if len(comparison_data) != len(player_names):
            found_players = comparison_data['player_name'].tolist()
            missing_players = [p for p in player_names if p not in found_players]
            print(f"⚠️ Some players not found: {missing_players}")
            print(f"✅ Comparing: {found_players}")
        
        # Labels for the radar chart
        labels = ['Passing/Creation (P)', 'Finishing/Shot Value (S)', 'Carrying/1v1 (C)', 
                 'Defending/Disruption (D)', 'Discipline (L)']
        
        # Number of variables we're plotting
        num_vars = len(labels)
        
        # Add the first label to the end to close the circle
        labels = labels + labels[:1]
        
        # Compute angle for each axis
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        # Create the radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # Plot data for each player
        for i, (_, player) in enumerate(comparison_data.iterrows()):
            # Extract the 5 component scores
            scores = player[['P_score', 'S_score', 'C_score', 'D_score', 'L_score']].tolist()
            
            # Add the first score to the end to close the circle
            scores = scores + scores[:1]
            
            # Plot data and format grid
            ax.plot(angles, scores, linewidth=2, linestyle='solid', 
                   label=player['player_name'], color=self.colors[i % len(self.colors)])
            ax.fill(angles, scores, color=self.colors[i % len(self.colors)], alpha=0.25)
        
        # Set title and formatting
        ax.set_title('Performance Score (PSCDL Score) Comparison', size=20, color='black', y=1.1)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines and labels
        ax.set_thetagrids(np.degrees(angles), labels)
        
        # Go through labels and adjust alignment
        for label, angle in zip(ax.get_xticklabels(), angles):
            if angle == 0:
                label.set_horizontalalignment('center')
            elif angle == np.pi/2:
                label.set_horizontalalignment('center')
            elif angle == np.pi:
                label.set_horizontalalignment('center')
            elif angle == 3*np.pi/2:
                label.set_horizontalalignment('center')
            else:
                label.set_horizontalalignment('center')

            # Adjust vertical alignment for better positioning
            if angle > np.pi/2 and angle < 3*np.pi/2:
                 label.set_verticalalignment('bottom')
            else:
                 label.set_verticalalignment('top')

        # Draw ylabels
        ax.set_rgrids([20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Radar comparison plot saved to: {save_path}")
        
        plt.show()
    
    def create_comparison_dashboard(self, recruitment_df: pd.DataFrame, scouting_targets: List[str], 
                                  save_path: Optional[str] = None):
        """
        Create comprehensive player comparison dashboard
        
        Args:
            recruitment_df (pd.DataFrame): Recruitment data with performance scores
            scouting_targets (List[str]): List of player names to compare
            save_path (Optional[str]): Path to save the plot
        """
        if not scouting_targets:
            print("❌ Please provide a list of players to compare")
            return
        
        # Filter data for ONLY the selected players
        comparison_data = recruitment_df[recruitment_df['player_name'].isin(scouting_targets)]
        
        if len(comparison_data) == 0:
            print("❌ No players found in the dataset")
            print("Available players:", recruitment_df['player_name'].head(10).tolist())
            return
        
        if len(comparison_data) != len(scouting_targets):
            found_players = comparison_data['player_name'].tolist()
            missing_players = [p for p in scouting_targets if p not in found_players]
            print(f"⚠️ Some players not found: {missing_players}")
            print(f"✅ Comparing: {found_players}")
        
        # Create comparison visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 1. Bar chart comparison
        specialist_areas = ['P_score', 'S_score', 'C_score', 'D_score', 'L_score']
        specialist_labels = ['Passing/Creation', 'Finishing/Shot', 'Carrying/1v1', 
                           'Defending/Disruption', 'Discipline']
        
        x = np.arange(len(specialist_labels))
        width = 0.8 / len(comparison_data)
        
        # Use distinct colors
        if len(comparison_data) <= len(self.colors):
            colors = self.colors[:len(comparison_data)]
        else:
            colors = self.colors + list(plt.cm.tab20(np.linspace(0, 1, len(comparison_data) - len(self.colors))))
        
        for i, (_, player) in enumerate(comparison_data.iterrows()):
            scores = [player[area] for area in specialist_areas]
            ax1.bar(x + i * width, scores, width, label=player['player_name'], 
                   color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Specialist Areas')
        ax1.set_ylabel('Score')
        ax1.set_title('Specialist Score Comparison', fontweight='bold')
        ax1.set_xticks(x + width * (len(comparison_data) - 1) / 2)
        ax1.set_xticklabels(specialist_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 100)
        
        # 2. Radar chart comparison
        ax2 = plt.subplot(122, projection='polar')
        
        # Define angles for radar chart
        angles = np.linspace(0, 2*np.pi, len(specialist_areas), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, (_, player) in enumerate(comparison_data.iterrows()):
            values = [player[area] for area in specialist_areas]
            values += values[:1]  # Complete the circle
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=player['player_name'], 
                    color=colors[i])
            ax2.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(['P', 'S', 'C', 'D', 'L'])
        ax2.set_ylim(0, 100)
        ax2.set_title('Performance Profile Comparison', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        plt.suptitle('PLAYER COMPARISON DASHBOARD (Designed by Michael Xu)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison dashboard saved to: {save_path}")
        
        plt.show()
        
        return comparison_data
    
    def create_enhanced_comparison_dashboard(self, recruitment_df: pd.DataFrame, scouting_targets: List[str], 
                                           save_path: Optional[str] = None):
        """
        Create enhanced comparison dashboard with radar area metrics
        
        Args:
            recruitment_df (pd.DataFrame): Recruitment data with performance scores and radar area
            scouting_targets (List[str]): List of player names to compare
            save_path (Optional[str]): Path to save the plot
        """
        if not scouting_targets:
            print("❌ Please provide a list of players to compare")
            return
        
        # Filter data for ONLY the selected players
        comparison_data = recruitment_df[recruitment_df['player_name'].isin(scouting_targets)]
        
        if len(comparison_data) == 0:
            print("❌ No players found in the dataset")
            return
        
        if len(comparison_data) != len(scouting_targets):
            found_players = comparison_data['player_name'].tolist()
            missing_players = [p for p in scouting_targets if p not in found_players]
            print(f"⚠️ Some players not found: {missing_players}")
            print(f"✅ Comparing: {found_players}")
        
        # Create comparison visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bar chart comparison
        specialist_areas = ['P_score', 'S_score', 'C_score', 'D_score', 'L_score']
        specialist_labels = ['Passing/Creation', 'Finishing/Shot', 'Carrying/1v1', 
                           'Defending/Disruption', 'Discipline']
        
        x = np.arange(len(specialist_labels))
        width = 0.8 / len(comparison_data)
        
        # Use distinct colors
        if len(comparison_data) <= len(self.colors):
            colors = self.colors[:len(comparison_data)]
        else:
            colors = self.colors + list(plt.cm.tab20(np.linspace(0, 1, len(comparison_data) - len(self.colors))))
        
        for i, (_, player) in enumerate(comparison_data.iterrows()):
            scores = [player[area] for area in specialist_areas]
            ax1.bar(x + i * width, scores, width, label=player['player_name'], 
                   color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Specialist Areas')
        ax1.set_ylabel('Score')
        ax1.set_title('Specialist Score Comparison', fontweight='bold')
        ax1.set_xticks(x + width * (len(comparison_data) - 1) / 2)
        ax1.set_xticklabels(specialist_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 100)
        
        # 2. Radar chart comparison
        ax2 = plt.subplot(222, projection='polar')
        
        angles = np.linspace(0, 2*np.pi, len(specialist_areas), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (_, player) in enumerate(comparison_data.iterrows()):
            values = [player[area] for area in specialist_areas]
            values += values[:1]
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=player['player_name'], 
                    color=colors[i])
            ax2.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(['P', 'S', 'C', 'D', 'L'])
        ax2.set_ylim(0, 100)
        ax2.set_title('Performance Profile Comparison', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        # 3. Performance vs Radar Area scatter plot
        for i, (_, player) in enumerate(comparison_data.iterrows()):
            ax3.scatter(player['performance_score'], player['radar_area_normalized'], 
                       color=colors[i], s=200, alpha=0.7, label=player['player_name'])
            ax3.annotate(player['player_name'], 
                        (player['performance_score'], player['radar_area_normalized']),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Performance Score')
        ax3.set_ylabel('Radar Area (Normalized)')
        ax3.set_title('Performance Score vs Radar Area', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Radar Area comparison bar chart
        player_names = comparison_data['player_name'].tolist()
        radar_areas = comparison_data['radar_area_normalized'].tolist()
        
        bars = ax4.bar(player_names, radar_areas, color=colors[:len(comparison_data)], alpha=0.8)
        ax4.set_ylabel('Radar Area (Normalized)')
        ax4.set_title('Overall Ability Coverage Comparison', fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, area in zip(bars, radar_areas):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{area:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('ENHANCED PLAYER COMPARISON DASHBOARD (Designed by Michael Xu)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Enhanced comparison dashboard saved to: {save_path}")
        
        plt.show()
        
        return comparison_data
    
    def create_specialist_analysis_plots(self, specialists: Dict[str, pd.DataFrame], 
                                       save_path: Optional[str] = None):
        """
        Create plots for specialist analysis
        
        Args:
            specialists (Dict[str, pd.DataFrame]): Dictionary of specialists by area
            save_path (Optional[str]): Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        specialist_areas = ['P_score', 'S_score', 'C_score', 'D_score']
        specialist_labels = ['Passing/Creation', 'Finishing/Shot Value', 'Carrying/1v1', 'Defending/Disruption']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (area, label) in enumerate(zip(specialist_areas, specialist_labels)):
            if area in specialists and len(specialists[area]) > 0:
                top_specialists = specialists[area].head(8)
                
                bars = axes[i].barh(top_specialists['player_name'], top_specialists[area], 
                                  color=colors[i], alpha=0.8)
                axes[i].set_title(f'Top {label} Specialists', fontweight='bold')
                axes[i].set_xlabel('Score')
                axes[i].grid(True, alpha=0.3, axis='x')
                
                # Add value labels
                for j, (bar, value) in enumerate(zip(bars, top_specialists[area])):
                    axes[i].text(value + 1, j, f'{value:.1f}', 
                               va='center', fontweight='bold')
            else:
                axes[i].text(0.5, 0.5, f'No {label} specialists found', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{label} Specialists', fontweight='bold')
        
        plt.suptitle('SPECIALIST ANALYSIS (PSCDL Framework)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Specialist analysis plots saved to: {save_path}")
        
        plt.show()


def main():
    """
    Main function to demonstrate visualization tools
    """
    print("⚽ Soccer Analytics Pipeline - Visualization Tools Module")
    print("=" * 70)
    
    print("📝 This module provides comprehensive visualization tools including:")
    print("   - Performance distribution analysis")
    print("   - Top performers comparison charts")
    print("   - Player comparison dashboards")
    print("   - Radar chart visualizations")
    print("   - Specialist analysis plots")
    print("   - Enhanced comparison tools with radar area metrics")
    print("\n💡 Usage:")
    print("   from visualization_tools import SoccerVisualizationTools")
    print("   viz = SoccerVisualizationTools()")
    print("   viz.create_comparison_dashboard(recruitment_df, player_names)")


if __name__ == "__main__":
    main()
