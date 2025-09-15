"""
Soccer Analytics Dashboard - Interactive Web Application

A Flask-based web application that provides an interactive dashboard for soccer analytics,
allowing users to explore player data, compare performances, and generate insights
using the PSCDL framework.

Author: Michael Xu
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import warnings
warnings.filterwarnings('ignore')

# Import our analytics modules
import importlib.util
import sys

def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load analytics modules
try:
    recruitment_analyzer = load_module("recruitment_analyzer", "04_recruitment_analyzer.py")
    RecruitmentAnalyzer = recruitment_analyzer.RecruitmentAnalyzer
except Exception as e:
    print(f"Warning: Could not load recruitment analyzer: {e}")
    RecruitmentAnalyzer = None

app = Flask(__name__)

class SoccerDashboard:
    """Main dashboard class for handling data and visualizations"""
    
    def __init__(self):
        self.data_loaded = False
        self.performance_df = None
        self.totals_df = None
        self.recruitment_df = None
        self.analyzer = None
        
    def load_data(self):
        """Load the soccer analytics data"""
        try:
            stats_dir = Path("Statistics")
            performance_file = stats_dir / "statistics_performance_scores.csv"
            totals_file = stats_dir / "statistics_totals.csv"
            
            if not performance_file.exists() or not totals_file.exists():
                return False, "Statistics files not found. Please run the pipeline first."
            
            self.performance_df = pd.read_csv(performance_file)
            self.totals_df = pd.read_csv(totals_file)
            
            if RecruitmentAnalyzer:
                self.analyzer = RecruitmentAnalyzer(self.performance_df, self.totals_df)
                self.recruitment_df = self.analyzer.recruitment_df
                # Add radar area metrics for scatter plot
                self.analyzer.add_radar_area_metric(verbose=False)
                self.recruitment_df = self.analyzer.recruitment_df
            else:
                # Create basic recruitment dataframe
                self.recruitment_df = self.performance_df.merge(
                    self.totals_df[['player_id', 'player_name', 'total_games', 'total_minutes_played']], 
                    on=['player_id', 'player_name'], 
                    how='left'
                )
                # Calculate radar area metrics manually for basic case
                self._add_radar_area_metric()
            
            self.data_loaded = True
            return True, "Data loaded successfully"
            
        except Exception as e:
            return False, f"Error loading data: {e}"
    
    def _add_radar_area_metric(self):
        """Add radar area metrics to recruitment_df for basic case"""
        def calculate_radar_area(scores):
            """Calculate the area of a radar chart polygon"""
            if len(scores) < 3:
                return 0
            
            # Convert to radians and calculate area using shoelace formula
            angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False)
            x = scores * np.cos(angles)
            y = scores * np.sin(angles)
            
            # Shoelace formula
            area = 0.5 * abs(sum(x[i] * y[(i + 1) % len(y)] - x[(i + 1) % len(x)] * y[i] 
                               for i in range(len(x))))
            return area
        
        # Calculate radar areas for each player
        radar_areas = []
        for _, player in self.recruitment_df.iterrows():
            scores = [player['P_score'], player['S_score'], player['C_score'], 
                     player['D_score'], player['L_score']]
            area = calculate_radar_area(scores)
            radar_areas.append(area)
        
        self.recruitment_df['radar_area'] = radar_areas
        
        # Normalize to 0-100 scale (theoretical max area for 100x100 square)
        theoretical_max_area = 0.5 * 100 * 100 * np.sin(2 * np.pi / 5)  # Regular pentagon
        self.recruitment_df['radar_area_normalized'] = (self.recruitment_df['radar_area'] / theoretical_max_area) * 100
    
    def get_performance_distribution_chart(self):
        """Create performance score distribution chart"""
        if not self.data_loaded:
            return {}
            
        # Create histogram data manually to avoid serialization issues
        scores = self.recruitment_df['performance_score'].dropna()
        
        # Calculate histogram
        hist, bin_edges = np.histogram(scores, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate statistics
        mean_score = scores.mean()
        top_10_threshold = scores.quantile(0.9)
        
        # Create traces
        data = [{
            'x': bin_centers.tolist(),
            'y': hist.tolist(),
            'type': 'bar',
            'name': 'Performance Score Distribution',
            'marker': {'color': '#636efa'},
            'hovertemplate': '<b>Score:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        }]
        
        # Add vertical lines
        shapes = [
            {
                'type': 'line',
                'x0': mean_score,
                'x1': mean_score,
                'y0': 0,
                'y1': 1,
                'yref': 'y domain',
                'line': {'color': 'red', 'dash': 'dash'}
            },
            {
                'type': 'line',
                'x0': top_10_threshold,
                'x1': top_10_threshold,
                'y0': 0,   
                'y1': 1,
                'yref': 'y domain',
                'line': {'color': 'orange', 'dash': 'dash'}
            }
        ]
        
        # Add annotations
        annotations = [
            {
                'x': mean_score,
                'y': 1,
                'xref': 'x',
                'yref': 'y domain',
                'text': f'<b>Mean: {mean_score:.1f}</b>',
                'showarrow': False,
                'xanchor': 'left',
                'yanchor': 'bottom'
            },
            {
                'x': top_10_threshold,
                'y': 1,
                'xref': 'x',
                'yref': 'y domain',
                'text': f'<b>Top 10%: {top_10_threshold:.1f}</b>',
                'showarrow': False,
                'xanchor': 'left',
                'yanchor': 'bottom'
            }
        ]
        
        layout = {
            'title': 'Performance Score Distribution',
            'xaxis': {'title': 'Performance Score'},
            'yaxis': {'title': 'Number of Players'},
            'height': 400,
            'shapes': shapes,
            'annotations': annotations
        }
        
        return json.dumps({'data': data, 'layout': layout})
    
    def get_top_specialists_chart(self, category='performance_score', limit=10):
        """Create top performers bar chart"""
        if not self.data_loaded:
            return {}
            
        qualified_players = self.recruitment_df[self.recruitment_df['total_games'] >= 5]
        top_players = qualified_players.nlargest(limit, category)
        
        # Create data manually
        data = [{
            'x': top_players[category].tolist(),
            'y': top_players['player_name'].tolist(),
            'type': 'bar',
            'orientation': 'h',
            'name': f'Top {limit} Players',
            'marker': {'color': '#636efa'},
            'text': round(top_players[category], 2).tolist(),
            'textposition': 'outside',
            'hovertemplate': '<b>%{y}</b><br><b>Score:</b> %{x}<extra></extra>'
        }]
        
        layout = {
            'title': f'Top {limit} Players by {category.replace("_", " ").title()}',
            'xaxis': {
                'title': category.replace('_', ' ').title(),
                'range': [0, 100]  # Set x-axis range to 0-100
            },
            'yaxis': {
                'title': 'Players', 
                'categoryorder': 'total ascending',
                'title_standoff': 100
            },
            'height': 400,
            'margin': {'l': 150, 'r': 30, 't': 30, 'b': 30}
        }
        
        return json.dumps({'data': data, 'layout': layout})
    
    def get_pscdl_radar_chart(self, player_names):
        """Create PSCDL radar chart for selected players"""
        if not self.data_loaded:
            return {}
            
        # Filter data for selected players
        selected_players = self.recruitment_df[self.recruitment_df['player_name'].isin(player_names)]
        
        if selected_players.empty:
            return {}
        
        # Create radar chart
        categories = ['P_score', 'S_score', 'C_score', 'D_score', 'L_score']
        category_labels = ['Passing/Creation', 'Finishing/Shot', 'Carrying/1v1', 'Defending/Disruption', 'Discipline']
        
        fig = go.Figure()
        
        # Use consistent colors with bar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (_, player) in enumerate(selected_players.iterrows()):
            values = [player[cat] for cat in categories]
            values += values[:1]  # Complete the circle
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=category_labels + [category_labels[0]],  # Complete the circle
                fill='toself',
                name=player['player_name'],
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.5
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="PSCDL Performance Comparison",
            height=500
        )
        
        return json.dumps(fig, cls=PlotlyJSONEncoder)
    
    def get_pscdl_bar_chart(self, player_names):
        """Create PSCDL bar chart for selected players"""
        if not self.data_loaded:
            return {}
            
        # Filter data for selected players
        selected_players = self.recruitment_df[self.recruitment_df['player_name'].isin(player_names)]
        
        if selected_players.empty:
            return {}
        
        # Create bar chart
        categories = ['P_score', 'S_score', 'C_score', 'D_score', 'L_score']
        category_labels = ['Passing/Creation', 'Finishing/Shot', 'Carrying/1v1', 'Defending/Disruption', 'Discipline']
        
        # Use consistent colors with radar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        traces = []
        for i, (_, player) in enumerate(selected_players.iterrows()):
            scores = [player[cat] for cat in categories]
            
            traces.append({
                'x': category_labels,
                'y': scores,
                'name': player['player_name'],
                'type': 'bar',
                'marker': {'color': colors[i % len(colors)]},
                'text': [f"{score:.1f}" for score in scores],
                'textposition': 'outside',
                'hovertemplate': '<b>%{fullData.name}</b><br><b>Area:</b> %{x}<br><b>Score:</b> %{y}<extra></extra>'
            })
        
        layout = {
            'title': 'Specialist Score Comparison',
            'xaxis': {'title': 'Specialist Areas', 'automargin': True},
            'yaxis': {'title': 'Score', 'range': [0, 100], 'automargin': True},
            'barmode': 'group',
            'height': 400,
            'margin': {'l': 100, 'r': 50, 't': 50, 'b': 50}
        }
        
        return json.dumps({'data': traces, 'layout': layout}, cls=PlotlyJSONEncoder)
    
    def get_scatter_plot(self, x_col='performance_score', y_col='radar_area_normalized'):
        """Create scatter plot for custom x and y axes"""
        if not self.data_loaded:
            return {}
            
        # Available metrics for scatter plot
        available_metrics = {
            'performance_score': 'PSCDL Performance Score',
            'P_score': 'Passing/Creation Score',
            'S_score': 'Finishing/Shot Value Score', 
            'C_score': 'Carrying/1v1 Score',
            'D_score': 'Defending/Disruption Score',
            'L_score': 'Discipline Score',
            'radar_area_normalized': 'Ability Coverage (Radar Area)',
            'total_games': 'Total Games Played',
            'total_minutes_played': 'Total Minutes Played'
        }
        
        # Validate columns exist
        if x_col not in self.recruitment_df.columns or y_col not in self.recruitment_df.columns:
            return {}
        
        # Create scatter plot data manually
        data = [{
            'x': self.recruitment_df[x_col].tolist(),
            'y': self.recruitment_df[y_col].tolist(),
            'mode': 'markers',
            'type': 'scatter',
            'name': 'Players',
            'marker': {'color': '#636efa', 'size': 8, 'opacity': 0.7},
            'text': self.recruitment_df['player_name'].tolist(),
            'hovertemplate': f'<b>%{{text}}</b><br>{available_metrics.get(x_col, x_col.replace("_", " ").title())}: %{{x}}<br>{available_metrics.get(y_col, y_col.replace("_", " ").title())}: %{{y}}<extra></extra>'
        }]
        
        layout = {
            'title': f'{available_metrics.get(x_col, x_col.replace("_", " ").title())} vs {available_metrics.get(y_col, y_col.replace("_", " ").title())}',
            'xaxis': {'title': available_metrics.get(x_col, x_col.replace('_', ' ').title())},
            'yaxis': {'title': available_metrics.get(y_col, y_col.replace('_', ' ').title())},
            'height': 500,
            'margin': {'l': 100, 'r': 50, 't': 50, 'b': 50}
        }
        
        return json.dumps({'data': data, 'layout': layout})
    
    def get_top_specialists_table(self, area='performance_score', limit=10):
        """Get specialist table data"""
        if not self.data_loaded:
            return []
            
        specialists = self.recruitment_df[self.recruitment_df['total_games'] >= 5].nlargest(limit, area)
        
        return specialists[['player_name', 'team', area, 'performance_score', 'total_games']].to_dict('records')
    
    def get_player_comparison_data(self, player_names):
        """Get detailed comparison data for selected players"""
        if not self.data_loaded:
            return []
            
        selected_players = self.recruitment_df[self.recruitment_df['player_name'].isin(player_names)]
        
        if selected_players.empty:
            return []
        
        # Select relevant columns for comparison
        comparison_cols = [
            'player_name', 'team', 'performance_score', 'P_score', 'S_score', 
            'C_score', 'D_score', 'L_score', 'total_games', 'total_minutes_played'
        ]
        
        return selected_players[comparison_cols].to_dict('records')
    
    def get_basic_stats_chart(self, stat_type='goals', limit=10):
        """Create basic statistics chart showing top performers in key stats"""
        if not self.data_loaded:
            return {}
            
        # Define all available statistics with their display names and colors
        all_stats = {
            'total_goals': {'name': 'Goals', 'color': '#FF6B6B'},
            'total_xg': {'name': 'Expected Goals (xG)', 'color': '#4ECDC4'},
            'total_assists': {'name': 'Assists', 'color': '#45B7D1'},
            'total_interceptions': {'name': 'Interceptions', 'color': '#96CEB4'},
            'total_tackles': {'name': 'Tackles', 'color': '#c80ee1'},
            'total_fouls_committed': {'name': 'Fouls Committed', 'color': '#FFEAA7'}
        }
        
        # Map stat types to columns
        stat_mapping = {
            'goals': 'total_goals',
            'xg': 'total_xg',
            'assists': 'total_assists',
            'interceptions': 'total_interceptions',
            'tackles': 'total_tackles',
            'fouls': 'total_fouls_committed'
        }
        
        traces = []
        
        # Show only selected statistic
        if stat_type in stat_mapping:
            stat_col = stat_mapping[stat_type]
            stat_info = all_stats[stat_col]
            
            if stat_col in self.totals_df.columns:
                # Get top players for this stat, sorted by highest values first
                top_players = self.totals_df.nlargest(limit, stat_col)
                
                # Filter out rows where the stat value is null or 0
                top_players = top_players[top_players[stat_col].notna() & (top_players[stat_col] > 0)]
                
                if len(top_players) > 0:
                    # Reverse the order to show highest values at the top
                    top_players = top_players.iloc[::-1]
                    
                    # Format text labels based on stat type
                    if stat_type == 'xg':
                        text_labels = [f"{val:.2f}" for val in top_players[stat_col].tolist()]
                    else:
                        text_labels = [f"{int(val)}" for val in top_players[stat_col].tolist()]
                    
                    traces.append({
                        'x': top_players[stat_col].tolist(),
                        'y': top_players['player_name'].tolist(),
                        'type': 'bar',
                        'orientation': 'h',
                        'name': stat_info['name'],
                        'marker': {'color': stat_info['color']},
                        'text': text_labels,
                        'textposition': 'outside',
                        'hovertemplate': '<b>%{y}</b><br><b>Count:</b> %{x}<extra></extra>'
                    })
        
        layout = {
            'title': f'Statistics Leaders - {all_stats[stat_mapping[stat_type]]["name"] if stat_type in stat_mapping else "Unknown"}',
            'xaxis': {'title': 'Count', 'automargin': True},
            'yaxis': {
                'automargin': True,
                'title_standoff': 100,  # Increase space between title and player names
                'tickmode': 'linear'
            },
            'height': 500,
            'margin': {'l': 200, 'r': 100, 't': 50, 'b': 50}  # Increase left margin for more space
        }
        
        return json.dumps({'data': traces, 'layout': layout}, cls=PlotlyJSONEncoder)

# Initialize dashboard
dashboard = SoccerDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/load_data')
def load_data():
    """API endpoint to load data"""
    success, message = dashboard.load_data()
    return jsonify({'success': success, 'message': message})

@app.route('/api/performance_distribution')
def performance_distribution():
    """API endpoint for performance distribution chart"""
    chart_data = dashboard.get_performance_distribution_chart()
    return chart_data

@app.route('/api/top_performers')
def top_performers():
    """API endpoint for top performers chart"""
    category = request.args.get('category', 'performance_score')
    limit = int(request.args.get('limit', 10))
    chart_data = dashboard.get_top_specialists_chart(category, limit)
    return chart_data

@app.route('/api/pscdl_radar')
def pscdl_radar():
    """API endpoint for PSCDL radar chart"""
    player_names = request.args.getlist('players')
    chart_data = dashboard.get_pscdl_radar_chart(player_names)
    return chart_data

@app.route('/api/pscdl_bar')
def pscdl_bar():
    """API endpoint for PSCDL bar chart"""
    player_names = request.args.getlist('players')
    chart_data = dashboard.get_pscdl_bar_chart(player_names)
    return chart_data

@app.route('/api/scatter_plot')
def scatter_plot():
    """API endpoint for scatter plot"""
    x_col = request.args.get('x', 'performance_score')
    y_col = request.args.get('y', 'radar_area_normalized')
    chart_data = dashboard.get_scatter_plot(x_col, y_col)
    return chart_data

@app.route('/api/specialists')
def specialists():
    """API endpoint for specialist table"""
    area = request.args.get('area', 'performance_score')
    limit = int(request.args.get('limit', 10))
    table_data = dashboard.get_top_specialists_table(area, limit)
    return jsonify(table_data)

@app.route('/api/player_comparison')
def player_comparison():
    """API endpoint for player comparison data"""
    player_names = request.args.getlist('players')
    comparison_data = dashboard.get_player_comparison_data(player_names)
    return jsonify(comparison_data)

@app.route('/api/players_list')
def players_list():
    """API endpoint to get list of all players"""
    if not dashboard.data_loaded:
        return jsonify([])
    
    players = dashboard.recruitment_df[['player_name', 'team', 'performance_score']].to_dict('records')
    return jsonify(players)

@app.route('/api/data_status')
def data_status():
    """API endpoint to check data loading status"""
    return jsonify({
        'loaded': dashboard.data_loaded,
        'player_count': len(dashboard.recruitment_df) if dashboard.data_loaded else 0
    })

@app.route('/api/basic_stats_chart')
def basic_stats_chart():
    """API endpoint for basic statistics chart"""
    stat_type = request.args.get('stat_type', 'all')
    limit = int(request.args.get('limit', 10))
    chart_data = dashboard.get_basic_stats_chart(stat_type, limit)
    return chart_data

#if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Create static directory if it doesn't exist
    static_dir = Path('static')
    static_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting Soccer Analytics Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:5000")
    print("💡 Make sure you have run the pipeline to generate data first!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)


