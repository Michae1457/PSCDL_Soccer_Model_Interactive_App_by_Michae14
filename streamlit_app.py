"""
Soccer Analytics Dashboard - Multi-League Streamlit Version

A Streamlit-based web application that provides an interactive dashboard for soccer analytics,
allowing users to explore player data from multiple leagues and competitions, compare performances, 
and generate insights using the PSCDL framework.

Author: Michael Xu
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import warnings
import os
from pathlib import Path
import json
import base64
warnings.filterwarnings('ignore')

# Import our analytics modules
import importlib.util
import sys

# Fixed competition data with stable indices
COMPETITION_DATA = {
    'La_Liga': {'display_name': 'La Liga', 'info': {'competition_id': 11, 'name': 'La Liga', 'country': 'Spain', 'number_of_seasons': 18, 'seasons': {'2020/2021': {'season_id': 90, 'number_of_matches': 35}, '2019/2020': {'season_id': 42, 'number_of_matches': 33}, '2018/2019': {'season_id': 4, 'number_of_matches': 34}, '2017/2018': {'season_id': 1, 'number_of_matches': 36}, '2016/2017': {'season_id': 2, 'number_of_matches': 34}, '2015/2016': {'season_id': 27, 'number_of_matches': 380}, '2014/2015': {'season_id': 26, 'number_of_matches': 38}, '2013/2014': {'season_id': 25, 'number_of_matches': 31}, '2012/2013': {'season_id': 24, 'number_of_matches': 32}, '2011/2012': {'season_id': 23, 'number_of_matches': 37}, '2010/2011': {'season_id': 22, 'number_of_matches': 33}, '2009/2010': {'season_id': 21, 'number_of_matches': 35}, '2008/2009': {'season_id': 41, 'number_of_matches': 31}, '2007/2008': {'season_id': 40, 'number_of_matches': 28}, '2006/2007': {'season_id': 39, 'number_of_matches': 26}, '2005/2006': {'season_id': 38, 'number_of_matches': 17}, '2004/2005': {'season_id': 37, 'number_of_matches': 7}, '1973/1974': {'season_id': 278, 'number_of_matches': 1}}}},
    'Premier_League': {'display_name': 'Premier League', 'info': {'competition_id': 2, 'name': 'Premier League', 'country': 'England', 'number_of_seasons': 2, 'seasons': {'2015/2016': {'season_id': 27, 'number_of_matches': 380}, '2003/2004': {'season_id': 44, 'number_of_matches': 38}}}},
    '1._Bundesliga': {'display_name': '1. Bundesliga', 'info': {'competition_id': 9, 'name': '1. Bundesliga', 'country': 'Germany', 'number_of_seasons': 2, 'seasons': {'2023/2024': {'season_id': 281, 'number_of_matches': 34}, '2015/2016': {'season_id': 27, 'number_of_matches': 306}}}},
    'Serie_A': {'display_name': 'Serie A', 'info': {'competition_id': 12, 'name': 'Serie A', 'country': 'Italy', 'number_of_seasons': 2, 'seasons': {'2015/2016': {'season_id': 27, 'number_of_matches': 380}, '1986/1987': {'season_id': 86, 'number_of_matches': 1}}}},
    'Ligue_1': {'display_name': 'Ligue 1', 'info': {'competition_id': 7, 'name': 'Ligue 1', 'country': 'France', 'number_of_seasons': 3, 'seasons': {'2022/2023': {'season_id': 235, 'number_of_matches': 32}, '2021/2022': {'season_id': 108, 'number_of_matches': 26}, '2015/2016': {'season_id': 27, 'number_of_matches': 377}}}},
    'FIFA_World_Cup': {'display_name': 'FIFA World Cup', 'info': {'competition_id': 43, 'name': 'FIFA World Cup', 'country': 'International', 'number_of_seasons': 8, 'seasons': {'2022': {'season_id': 106, 'number_of_matches': 64}, '2018': {'season_id': 3, 'number_of_matches': 64}, '1990': {'season_id': 55, 'number_of_matches': 1}, '1986': {'season_id': 54, 'number_of_matches': 3}, '1974': {'season_id': 51, 'number_of_matches': 6}, '1970': {'season_id': 272, 'number_of_matches': 6}, '1962': {'season_id': 270, 'number_of_matches': 1}, '1958': {'season_id': 269, 'number_of_matches': 2}}}},
    'UEFA_Euro': {'display_name': 'UEFA Euro', 'info': {'competition_id': 55, 'name': 'UEFA Euro', 'country': 'Europe', 'number_of_seasons': 2, 'seasons': {'2024': {'season_id': 282, 'number_of_matches': 51}, '2020': {'season_id': 43, 'number_of_matches': 51}}}},
    'Copa_America': {'display_name': 'Copa America', 'info': {'competition_id': 223, 'name': 'Copa America', 'country': 'South America', 'number_of_seasons': 1, 'seasons': {'2024': {'season_id': 282, 'number_of_matches': 32}}}},
    'African_Cup_of_Nations': {'display_name': 'African Cup of Nations', 'info': {'competition_id': 1267, 'name': 'African Cup of Nations', 'country': 'Africa', 'number_of_seasons': 1, 'seasons': {'2023': {'season_id': 107, 'number_of_matches': 52}}}},
    "Women's_World_Cup": {'display_name': "Women's World Cup", 'info': {'competition_id': 72, 'name': "Women's World Cup", 'country': 'International', 'number_of_seasons': 2, 'seasons': {'2023': {'season_id': 107, 'number_of_matches': 64}, '2019': {'season_id': 30, 'number_of_matches': 52}}}},
    "UEFA_Women's_Euro": {'display_name': "UEFA Women's Euro", 'info': {'competition_id': 53, 'name': "UEFA Women's Euro", 'country': 'Europe', 'number_of_seasons': 2, 'seasons': {'2025': {'season_id': 315, 'number_of_matches': 31}, '2022': {'season_id': 106, 'number_of_matches': 31}}}},
    "FA_Women's_Super_League": {'display_name': "FA Women's Super League", 'info': {'competition_id': 37, 'name': "FA Women's Super League", 'country': 'England', 'number_of_seasons': 3, 'seasons': {'2020/2021': {'season_id': 90, 'number_of_matches': 131}, '2019/2020': {'season_id': 42, 'number_of_matches': 87}, '2018/2019': {'season_id': 4, 'number_of_matches': 108}}}}
}

# Customizable order - you can change this to reorder leagues
COMPETITION_ORDER = [
    'La_Liga',           # 0 - Top European League
    'Premier_League',    # 1 - Top European League  
    '1._Bundesliga',     # 2 - Top European League
    'Serie_A',           # 3 - Top European League
    'Ligue_1',           # 4 - Top European League
    'FIFA_World_Cup',    # 5 - Major International
    'UEFA_Euro',         # 6 - Major International
    'Copa_America',      # 7 - Major International
    'African_Cup_of_Nations', # 8 - Major International
    "Women's_World_Cup",  # 9 - Women's Major
    "UEFA_Women's_Euro", # 10 - Women's Major
    "FA_Women's_Super_League" # 11 - Women's Major
]

# Generate available_competitions with fixed indices
available_competitions = []
for i, folder_name in enumerate(COMPETITION_ORDER):
    if folder_name in COMPETITION_DATA:
        comp_data = COMPETITION_DATA[folder_name]
        available_competitions.append({
            'folder_name': folder_name,
            'display_name': comp_data['display_name'],
            'info': comp_data['info'],
            'fixed_index': i  # Add fixed index for stability
        })


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load analytics modules
try:
    recruitment_analyzer = load_module("recruitment_analyzer", "Pipeline/04_recruitment_analyzer.py")
    RecruitmentAnalyzer = recruitment_analyzer.RecruitmentAnalyzer
except Exception as e:
    st.warning(f"Could not load recruitment analyzer: {e}")
    RecruitmentAnalyzer = None

# Load competition data
try:
    competitions_id = load_module("competitions_id", "Pipeline/utils/competitions_id.py")
    COMPETITION_INFO = competitions_id.COMPETITION_INFO
except Exception as e:
    st.error(f"Could not load competition data: {e}")
    COMPETITION_INFO = {}

# Page configuration
st.set_page_config(
    page_title="Soccer Analytics Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force light theme regardless of system settings
st.markdown("""
<script>
    // Force light theme and override system settings
    const setLightTheme = () => {
        // Override Streamlit's theme detection
        const root = document.documentElement;
        root.setAttribute('data-theme', 'light');
        
        // Force light theme colors
        root.style.setProperty('--background-color', '#FFFFFF');
        root.style.setProperty('--secondary-background-color', '#F0F8F0');
        root.style.setProperty('--text-color', '#262730');
        root.style.setProperty('--primary-color', '#2E8B57');
        
        // Override any dark theme classes
        document.body.classList.remove('dark');
        document.body.classList.add('light');
        
        // Force Streamlit to use light theme
        if (window.parent !== window) {
            window.parent.postMessage({
                type: 'streamlit:setThemeConfig',
                themeConfig: {
                    base: 'light',
                    primaryColor: '#2E8B57',
                    backgroundColor: '#FFFFFF',
                    secondaryBackgroundColor: '#F0F8F0',
                    textColor: '#262730'
                }
            }, '*');
        }
    };
    
    // Apply theme immediately and on any changes
    setLightTheme();
    document.addEventListener('DOMContentLoaded', setLightTheme);
    window.addEventListener('load', setLightTheme);
    
    // Override theme detection
    const observer = new MutationObserver(() => {
        setLightTheme();
    });
    observer.observe(document.body, { attributes: true, childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Custom CSS for green theme with white background - Force light theme
st.markdown("""
<style>
    /* Force light theme - override any dark theme styles */
    :root {
        --background-color: #FFFFFF !important;
        --secondary-background-color: #F0F8F0 !important;
        --text-color: #262730 !important;
        --primary-color: #2E8B57 !important;
    }
    
    /* Set overall page background to white - force override */
    .main .block-container {
        background-color: white !important;
    }
    
    .stApp {
        background-color: white !important;
    }
    
    /* Override dark theme styles */
    .stApp[data-theme="dark"],
    .stApp[data-theme="light"],
    .stApp {
        background-color: white !important;
    }
    
    /* Force light theme on all elements */
    .main .block-container,
    .main .block-container * {
        background-color: white !important;
        color: #262730 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E8B57, #32CD32, #90EE90);
        padding: 2rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-header h1,
    .main-header p,
    .main-header * {
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2E8B57, #32CD32);
        padding: 1rem;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-card h2,
    .metric-card p,
    .metric-card * {
        color: white !important;
    }
    
    .section-header {
        background: linear-gradient(135deg, #2E8B57, #32CD32);
        padding: 1rem;
        border-radius: 8px;
        color: white !important;
        margin: 1rem 0;
    }
    
    .section-header h3,
    .section-header i,
    .section-header * {
        color: white !important;
    }
    
    .competition-card {
        background: white;
        border: 2px solid #2E8B57;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .competition-card:hover {
        background: #F0F8F0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(46, 139, 87, 0.2);
    }
    
    .competition-logo {
        max-width: 120px;
        max-height: 120px;
        margin: 0 auto 1rem;
        display: block;
    }
    
    .competition-info {
        background: #F0F8F0;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: left;
    }
    
    .season-item {
        background: white;
        border: 1px solid #2E8B57;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .select-button {
        background: linear-gradient(135deg, #2E8B57, #32CD32);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        cursor: pointer;
        margin-top: 1rem;
        width: 100%;
    }
    
    .select-button:hover {
        background: linear-gradient(135deg, #228B22, #00FF00);
    }
    
    .stSelectbox > div > div {
        background-color: #f0f8f0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57, #32CD32);
        color: white;
        border: none;
        border-radius: 5px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #228B22, #00FF00);
        color: white;
    }
    
    /* Ensure all text is dark on white background */
    .main .block-container p, 
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3, 
    .main .block-container h4, 
    .main .block-container h5, 
    .main .block-container h6 {
        color: #262730;
    }
    
    /* Style the sidebar if needed */
    .css-1d391kg {
        background-color: #f8f9fa !important;
    }
    
    /* Override dark theme specific elements */
    .stSelectbox > div > div,
    .stSelectbox > div > div > div,
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stDataFrame,
    .stDataFrame > div,
    .stDataFrame table,
    .stDataFrame th,
    .stDataFrame td {
        background-color: white !important;
        color: #262730 !important;
        border-color: #e0e0e0 !important;
    }
    
    /* Override any dark theme text colors */
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown h1,
    .stMarkdown h2,
    .stMarkdown h3,
    .stMarkdown h4,
    .stMarkdown h5,
    .stMarkdown h6,
    .stMarkdown li,
    .stMarkdown ul,
    .stMarkdown ol {
        color: #262730 !important;
    }
    
    /* Override dark theme for specific Streamlit components */
    .stAlert,
    .stAlert > div,
    .stInfo,
    .stInfo > div,
    .stSuccess,
    .stSuccess > div,
    .stWarning,
    .stWarning > div,
    .stError,
    .stError > div {
        background-color: #f0f8f0 !important;
        color: #262730 !important;
        border-color: #2E8B57 !important;
    }
    
    /* Force light theme on all Streamlit widgets */
    .stWidget > div,
    .stWidget > div > div,
    .stWidget label,
    .stWidget .stMarkdown {
        background-color: white !important;
        color: #262730 !important;
    }
    
    /* Override any conflicting text colors in green sections */
    .main-header,
    .main-header *,
    .metric-card,
    .metric-card *,
    .section-header,
    .section-header * {
        color: white !important;
    }
    
    /* Specific overrides for green section text */
    .main-header h1,
    .main-header p,
    .main-header span,
    .main-header div,
    .metric-card h2,
    .metric-card p,
    .metric-card span,
    .metric-card div,
    .section-header h3,
    .section-header p,
    .section-header span,
    .section-header div,
    .section-header i {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

def get_competition_seasons(comp_folder_name):
    """Get available seasons for a competition"""
    stats_dir = Path("Statistics") / comp_folder_name
    seasons = []
    
    if not stats_dir.exists():
        return seasons
    
    for season_dir in stats_dir.iterdir():
        if season_dir.is_dir() and list(season_dir.glob("*.csv")):
            seasons.append(season_dir.name)
    
    return sorted(seasons)

def find_logo_file(comp_folder_name):
    """Find logo file for a competition, checking multiple extensions"""
    logo_dir = Path("Logo")
    if not logo_dir.exists():
        return None
    
    # Try different extensions
    extensions = ['.jpg']
    base_name = f"{comp_folder_name}_Logo"
    
    for ext in extensions:
        logo_path = logo_dir / f"{base_name}{ext}"
        if logo_path.exists():
            return logo_path
    
    return None


@st.cache_data
def load_competition_data(comp_folder_name, season="all", season_folder="all_seasons"):
    """Load data for a specific competition and season"""
    try:
        stats_dir = Path("Statistics") / comp_folder_name / season_folder
        
        if not stats_dir.exists():
            return None, None, None, f"No data found for {comp_folder_name} - {season}"
        
        # Look for the specific files
        performance_file = stats_dir / f"statistics_performance_scores_{comp_folder_name}_{season}.csv"
        totals_file = stats_dir / f"statistics_totals_{comp_folder_name}_{season}.csv"
        
        if not performance_file.exists() or not totals_file.exists():
            return None, None, None, f"Statistics files not found for {comp_folder_name} - {season}"
        
        performance_df = pd.read_csv(performance_file)
        totals_df = pd.read_csv(totals_file)
        
        if RecruitmentAnalyzer:
            analyzer = RecruitmentAnalyzer(performance_df, totals_df)
            recruitment_df = analyzer.recruitment_df
            # Add radar area metrics for scatter plot
            analyzer.add_radar_area_metric(verbose=False)
            recruitment_df = analyzer.recruitment_df
        else:
            # Create basic recruitment dataframe
            recruitment_df = performance_df.merge(
                totals_df[['player_id', 'player_name', 'total_games', 'total_minutes_played']], 
                on=['player_id', 'player_name'], 
                how='left'
            )
            # Calculate radar area metrics manually for basic case
            recruitment_df = add_radar_area_metric(recruitment_df)
        
        return performance_df, totals_df, recruitment_df, "Data loaded successfully"
        
    except Exception as e:
        return None, None, None, f"Error loading data: {e}"

def add_radar_area_metric(df):
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
    for _, player in df.iterrows():
        scores = [player['P_score'], player['S_score'], player['C_score'], 
                 player['D_score'], player['L_score']]
        area = calculate_radar_area(scores)
        radar_areas.append(area)
    
    df['radar_area'] = radar_areas
    
    # Normalize to 0-100 scale (theoretical max area for 100x100 square)
    theoretical_max_area = 0.5 * 100 * 100 * np.sin(2 * np.pi / 5)  # Regular pentagon
    df['radar_area_normalized'] = (df['radar_area'] / theoretical_max_area) * 100
    
    return df

def create_performance_distribution_chart(recruitment_df):
    """Create performance score distribution chart"""
    scores = recruitment_df['performance_score'].dropna()
    
    # Calculate histogram
    hist, bin_edges = np.histogram(scores, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate statistics
    mean_score = scores.mean()
    top_10_threshold = scores.quantile(0.9)
    
    fig = go.Figure()
    
    # Add histogram bars
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=hist,
        name='Performance Score Distribution',
        marker_color='#636efa',
        hovertemplate='<b>Score:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ))
    
    # Add vertical lines with annotations at the top
    fig.add_vline(x=mean_score, line_dash="dash", line_color="red", 
                  annotation_text=f"<b style='color: black;'>Mean: {mean_score:.1f}</b>",
                  annotation_position="top")
    fig.add_vline(x=top_10_threshold, line_dash="dash", line_color="orange", 
                  annotation_text=f"<b style='color: black;'>Top 10%: {top_10_threshold:.1f}</b>",
                  annotation_position="top")
    
    fig.update_layout(
        title='Performance Score Distribution',
        xaxis_title='Performance Score',
        yaxis_title='Number of Players',
        height=500,
        showlegend=False
    )
    
    return fig

def create_top_specialists_chart(recruitment_df, category='performance_score', limit=10):
    """Create top performers bar chart"""
    qualified_players = recruitment_df[recruitment_df['total_games'] >= 5]
    top_players = qualified_players.nlargest(limit, category)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_players[category],
        y=top_players['player_name'],
        orientation='h',
        name=f'Top {limit} Players',
        marker_color='#636efa',
        text=[f"{val:.1f}" for val in top_players[category]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br><b>Score:</b> %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Top {limit} Players by {category.replace("_", " ").title()}',
        xaxis_title=category.replace('_', ' ').title(),
        yaxis_title='Players',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(categoryorder='total ascending'),
        height=400,
        margin=dict(l=150, r=30, t=30, b=30)
    )
    
    return fig

def create_pscdl_radar_chart(recruitment_df, player_names):
    """Create PSCDL radar chart for selected players"""
    selected_players = recruitment_df[recruitment_df['player_name'].isin(player_names)]
    
    if selected_players.empty:
        return go.Figure()
    
    categories = ['P_score', 'S_score', 'C_score', 'D_score', 'L_score']
    category_labels = ['Passing/Creation', 'Finishing/Shot', 'Carrying/1v1', 'Defending/Disruption', 'Discipline']
    
    fig = go.Figure()
    
    # Use consistent colors
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
    
    return fig

def create_pscdl_bar_chart(recruitment_df, player_names):
    """Create PSCDL bar chart for selected players"""
    selected_players = recruitment_df[recruitment_df['player_name'].isin(player_names)]
    
    if selected_players.empty:
        return go.Figure()
    
    categories = ['P_score', 'S_score', 'C_score', 'D_score', 'L_score']
    category_labels = ['Passing/Creation', 'Finishing/Shot', 'Carrying/1v1', 'Defending/Disruption', 'Discipline']
    
    # Use consistent colors
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    fig = go.Figure()
    
    for i, (_, player) in enumerate(selected_players.iterrows()):
        scores = [player[cat] for cat in categories]
        
        fig.add_trace(go.Bar(
            x=category_labels,
            y=scores,
            name=player['player_name'],
            marker_color=colors[i % len(colors)],
            text=[f"{score:.1f}" for score in scores],
            textposition='outside',
            hovertemplate='<b>%{fullData.name}</b><br><b>Area:</b> %{x}<br><b>Score:</b> %{y}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Specialist Score Comparison',
        xaxis_title='Specialist Areas',
        yaxis_title='Score',
        yaxis=dict(range=[0, 100]),
        barmode='group',
        height=500,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig

def create_scatter_plot(recruitment_df, x_col='performance_score', y_col='radar_area_normalized'):
    """Create scatter plot for custom x and y axes"""
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
    if x_col not in recruitment_df.columns or y_col not in recruitment_df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recruitment_df[x_col],
        y=recruitment_df[y_col],
        mode='markers',
        name='Players',
        marker=dict(color='#636efa', size=8, opacity=0.7),
        text=recruitment_df['player_name'],
        hovertemplate=f'<b>%{{text}}</b><br>{available_metrics.get(x_col, x_col.replace("_", " ").title())}: %{{x}}<br>{available_metrics.get(y_col, y_col.replace("_", " ").title())}: %{{y}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{available_metrics.get(x_col, x_col.replace("_", " ").title())} vs {available_metrics.get(y_col, y_col.replace("_", " ").title())}',
        xaxis_title=available_metrics.get(x_col, x_col.replace('_', ' ').title()),
        yaxis_title=available_metrics.get(y_col, y_col.replace('_', ' ').title()),
        height=500,
        margin=dict(l=100, r=50, t=50, b=50)
    )
    
    return fig

def create_basic_stats_chart(totals_df, stat_type='goals', limit=10):
    """Create basic statistics chart showing top performers in key stats"""
    all_stats = {
        'total_goals': {'name': 'Goals', 'color': '#FF6B6B'},
        'total_xg': {'name': 'Expected Goals (xG)', 'color': '#4ECDC4'},
        'total_assists': {'name': 'Assists', 'color': '#45B7D1'},
        'total_interceptions': {'name': 'Interceptions', 'color': '#96CEB4'},
        'total_tackles': {'name': 'Tackles', 'color': '#c80ee1'},
        'total_fouls_committed': {'name': 'Fouls Committed', 'color': '#FFEAA7'}
    }
    
    stat_mapping = {
        'goals': 'total_goals',
        'xg': 'total_xg',
        'assists': 'total_assists',
        'interceptions': 'total_interceptions',
        'tackles': 'total_tackles',
        'fouls': 'total_fouls_committed'
    }
    
    if stat_type not in stat_mapping:
        return go.Figure()
    
    stat_col = stat_mapping[stat_type]
    stat_info = all_stats[stat_col]
    
    if stat_col not in totals_df.columns:
        return go.Figure()
    
    # Get top players for this stat
    top_players = totals_df.nlargest(limit, stat_col)
    top_players = top_players[top_players[stat_col].notna() & (top_players[stat_col] > 0)]
    
    if len(top_players) == 0:
        return go.Figure()
    
    # Reverse the order to show highest values at the top
    top_players = top_players.iloc[::-1]
    
    # Format text labels based on stat type
    if stat_type == 'xg':
        text_labels = [f"{val:.2f}" for val in top_players[stat_col].tolist()]
    else:
        text_labels = [f"{int(val)}" for val in top_players[stat_col].tolist()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_players[stat_col],
        y=top_players['player_name'],
        orientation='h',
        name=stat_info['name'],
        marker_color=stat_info['color'],
        text=text_labels,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br><b>Count:</b> %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'Statistics Leaders - {stat_info["name"]}',
        xaxis_title='Count',
        height=600,
        margin=dict(l=200, r=100, t=50, b=50)
    )
    
    return fig

def show_index_page():
    """Show the index page with competition selection"""
    
    # Access the global available_competitions list
    global available_competitions
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1><i class="fas fa-futbol"></i> Soccer Analytics Dashboard</h1>
        <p style="margin: 0; font-size: 1.2em;">Multi-League Player Performance Analysis</p>
        <p style="margin: 0; font-size: 1.2em;">Designed by Michael Xu</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PSCDL Framework Introduction
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-chart-line"></i> ⚽️ PSCDL Scoring Framework</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **PSCDL scoring score** is a 5-subscore performance framework that evaluates players across five critical dimensions: 
    **Passing/Creation (P)**, **Finishing/Shot Value (S)**, **Carrying/1v1 (C)**, **Defending/Disruption (D)**, and **Discipline (L)**. 
    These metrics are highly relevant for scouting because they capture the essential skills needed in modern soccer. 
    The framework includes 40+ statistical measures such as pass accuracy, progressive passes, xG per shot, successful dribbles, 
    tackle success rates, and interceptions per 90 minutes. **This standardized approach allows scouts to identify both specialists 
    (elite performers in specific areas) and all-rounders (balanced across multiple facets), enabling targeted recruitment based on tactical needs**.
    """)
    
    # Competition Selection Section
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-trophy"></i> 🏆 Select Competition to Analyze</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not available_competitions:
        st.error("No competition data found. Please run the pipeline first.")
        return
    
    # Create logo-based competition selection
    st.markdown("**Click on any competition section to view details and select:**")
    
    # Initialize session state for expanded competition
    if 'expanded_competition' not in st.session_state:
        st.session_state.expanded_competition = None
    
    # Create a grid of clickable logo buttons
    cols_per_row = 3
    rows = (len(available_competitions) + cols_per_row - 1) // cols_per_row
    
    for start_idx in range(0, len(available_competitions), cols_per_row):
        row_competitions = available_competitions[start_idx:start_idx + cols_per_row]
        if not row_competitions:
            continue
        
        row_cols = st.columns(len(row_competitions))
        
        for col_idx, comp in enumerate(row_competitions):
            # Safety check to prevent index errors
            #if col_idx >= len(cols):
                #st.error(f"Column index {col_idx} out of bounds for {len(cols)} columns")
                #continue
            
            # Use fixed index for session state keys to prevent corruption
            comp_key = f"comp_{comp['fixed_index']}"
                
            with row_cols[col_idx]:
                    # Create clickable logo button
                    logo_path = find_logo_file(comp['folder_name'])
                    
                    if logo_path:
                        # Create a styled container with logo inside
                        is_expanded = st.session_state.expanded_competition == comp_key
                        border_color = '#2E8B57' if is_expanded else '#E0E0E0'
                        bg_color = '#F0F8F0' if is_expanded else 'white'
                        text_color = '#2E8B57' if is_expanded else '#333'
                        
                        # Read logo and encode to base64
                        with open(logo_path, "rb") as img_file:
                            img_base64 = base64.b64encode(img_file.read()).decode()
                        
                        # Create the box with ONLY the logo inside (bigger)
                        st.markdown(f"""
                        <div style="border: 2px solid #E0E0E0; 
                                     border-radius: 20px; 
                                     padding: 25px; 
                                     background-color: white; 
                                     text-align: center; 
                                     min-height: 200px;
                                     margin: 10px 5px;
                                     display: flex;
                                     justify-content: center;
                                     align-items: center;">
                            <img src="data:image/jpeg;base64,{img_base64}" 
                                 style="width: 200px; height: 200px; object-fit: contain;">
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Competition name below the box
                        st.markdown(f"""
                        <div style="text-align: center; margin: 5px 0;">
                            <h4 style="margin: 0; color: #333; font-size: 18px; font-weight: bold;">
                                {comp['display_name']}
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        
                        # Check if this competition is selected for info expansion
                        is_expanded = st.session_state.expanded_competition == comp_key
                        
                        # Select button to show competition info
                        if st.button(f"Select", 
                                   key=f"select_{comp['folder_name']}", 
                                   use_container_width=True):
                            if is_expanded:
                                # Already expanded, collapse it
                                st.session_state.expanded_competition = None
                            else:
                                # Expand this competition, collapse others
                                st.session_state.expanded_competition = comp_key
                            st.rerun()
                        
                        # Show detailed competition info only if expanded
                        if is_expanded:
                            st.markdown("---")
                            total_matches = sum(season['number_of_matches'] for season in comp['info']['seasons'].values())
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin: 10px 0; color: #333; font-size: 15px;">
                                <p style="margin: 5px 0;"><strong>🏆 Competition:</strong> {comp['display_name']}</p>
                                <p style="margin: 5px 0;"><strong>🌍 Country:</strong> {comp['info']['country']}</p>
                                <p style="margin: 5px 0;"><strong>📅 Total Seasons:</strong> {comp['info']['number_of_seasons']}</p>
                                <p style="margin: 5px 0;"><strong>⚽ Total Matches:</strong> {total_matches}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detailed season information
                            st.markdown(f"""
                            <div style="text-align: center; margin: 10px 0;">
                                <h5 style="margin: 10px 0; color: #2E8B57;">📊 Season Details:</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show each season with match counts using Streamlit components
                            seasons_list = list(comp['info']['seasons'].items())
                            
                            if len(seasons_list) <= 3:
                                # Show all seasons in a single row if 3 or fewer
                                season_cols = st.columns(len(seasons_list))
                                for i, (season_name, season_data) in enumerate(seasons_list):
                                    with season_cols[i]:
                                        st.markdown(f"""
                                        <div style="background-color: #f0f8f0; border-radius: 10px; padding: 15px; margin: 5px; text-align: center;">
                                            <p style="margin: 5px 0; font-weight: bold; color: #333; font-size: 16px;">{season_name}</p>
                                            <p style="margin: 5px 0; color: #666; font-size: 14px;">{season_data['number_of_matches']} matches</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                # Show seasons in multiple rows for competitions with many seasons
                                season_cols = st.columns(3)  # Up to 3 seasons per row
                                for i, (season_name, season_data) in enumerate(seasons_list):
                                    with season_cols[i % 3]:
                                        st.markdown(f"""
                                        <div style="background-color: #f0f8f0; border-radius: 10px; padding: 15px; margin: 5px; text-align: center;">
                                            <p style="margin: 5px 0; font-weight: bold; color: #333; font-size: 16px;">{season_name}</p>
                                            <p style="margin: 5px 0; color: #666; font-size: 14px;">{season_data['number_of_matches']} matches</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Go to Dashboard button
                            if st.button(f"🎯 Go to Dashboard", 
                                       key=f"dashboard_{comp['folder_name']}", 
                                       use_container_width=True):
                                st.session_state.selected_competition = comp
                                st.session_state.current_page = "dashboard"
                                st.rerun()
                    else:
                        # Fallback for missing logos
                        st.markdown(f"""
                        <div style="border: 2px dashed #2E8B57; 
                                     border-radius: 20px; 
                                     padding: 25px; 
                                     background-color: white; 
                                     text-align: center; 
                                     min-height: 200px;
                                     margin: 10px 5px;
                                     display: flex;
                                     justify-content: center;
                                     align-items: center;">
                            <div>
                                <h4 style="color: #666;">{comp['display_name']}</h4>
                                <p style="color: #999; font-size: 14px;">Logo not found</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Competition name and info for missing logos
                        st.markdown(f"""
                        <div style="text-align: center; margin: 5px 0;">
                            <h4 style="margin: 0; color: #333; font-size: 18px; font-weight: bold;">
                                {comp['display_name']}
                            </h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Select button for missing logos
                        fallback_is_expanded = st.session_state.expanded_competition == comp_key
                        
                        if st.button(f"Select", 
                                   key=f"fallback_select_{comp['folder_name']}", 
                                   use_container_width=True):
                            if fallback_is_expanded:
                                st.session_state.expanded_competition = None
                            else:
                                st.session_state.expanded_competition = comp_key
                            st.rerun()
                        
                        # Show detailed competition info only if expanded (fallback)
                        if fallback_is_expanded:
                            st.markdown("---")
                            total_matches = sum(season['number_of_matches'] for season in comp['info']['seasons'].values())
                            
                            st.markdown(f"""
                            <div style="text-align: center; margin: 10px 0; color: #333; font-size: 15px;">
                                <p style="margin: 5px 0;"><strong>🏆 Competition:</strong> {comp['display_name']}</p>
                                <p style="margin: 5px 0;"><strong>🌍 Country:</strong> {comp['info']['country']}</p>
                                <p style="margin: 5px 0;"><strong>📅 Total Seasons:</strong> {comp['info']['number_of_seasons']}</p>
                                <p style="margin: 5px 0;"><strong>⚽ Total Matches:</strong> {total_matches}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Detailed season information (fallback)
                            st.markdown(f"""
                            <div style="text-align: center; margin: 10px 0;">
                                <h5 style="margin: 10px 0; color: #2E8B57;">📊 Season Details:</h5>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show each season with match counts (fallback) using Streamlit components
                            seasons_list = list(comp['info']['seasons'].items())
                            
                            if len(seasons_list) <= 3:
                                # Show all seasons in a single row if 3 or fewer
                                season_cols = st.columns(len(seasons_list))
                                for i, (season_name, season_data) in enumerate(seasons_list):
                                    with season_cols[i]:
                                        st.markdown(f"""
                                        <div style="background-color: #f0f8f0; border-radius: 10px; padding: 15px; margin: 5px; text-align: center;">
                                            <p style="margin: 5px 0; font-weight: bold; color: #333; font-size: 16px;">{season_name}</p>
                                            <p style="margin: 5px 0; color: #666; font-size: 14px;">{season_data['number_of_matches']} matches</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            else:
                                # Show seasons in multiple rows for competitions with many seasons
                                season_cols = st.columns(3)  # Up to 3 seasons per row
                                for i, (season_name, season_data) in enumerate(seasons_list):
                                    with season_cols[i % 3]:
                                        st.markdown(f"""
                                        <div style="background-color: #f0f8f0; border-radius: 10px; padding: 15px; margin: 5px; text-align: center;">
                                            <p style="margin: 5px 0; font-weight: bold; color: #333; font-size: 16px;">{season_name}</p>
                                            <p style="margin: 5px 0; color: #666; font-size: 14px;">{season_data['number_of_matches']} matches</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                            
                            # Go to Dashboard button (fallback)
                            if st.button(f"🎯 Go to Dashboard", 
                                       key=f"fallback_dashboard_{comp['folder_name']}", 
                                       use_container_width=True):
                                st.session_state.selected_competition = comp
                                st.session_state.current_page = "dashboard"
                                st.rerun()

def show_dashboard_page():
    """Show the dashboard page for selected competition"""
    
    if 'selected_competition' not in st.session_state:
        st.error("No competition selected. Please go back to the index page.")
        if st.button("Back to Index"):
            st.session_state.current_page = "index"
            st.rerun()
        return
    
    comp = st.session_state.selected_competition
    
    # Header with competition info
    st.markdown(f"""
    <div class="main-header">
        <h1><i class="fas fa-futbol"></i> {comp['display_name']} Analytics Dashboard</h1>
        <p style="margin: 0; font-size: 1.2em;">{comp['info']['country']} • {comp['info']['number_of_seasons']} Seasons Available</p>
        <p style="margin: 0; font-size: 1.2em;">Designed by Michael Xu</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button
    if st.button("← Back to Competition Selection"):
        st.session_state.current_page = "index"
        st.rerun()
    
    # Season selection
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-calendar"></i> 📅 Select Season</h3>
    </div>
    """, unsafe_allow_html=True)
    
    available_seasons = get_competition_seasons(comp['folder_name'])
    
    if not available_seasons:
        st.error(f"No season data found for {comp['display_name']}")
        return
    
    # Add "All Seasons" option
    season_options = ["All Seasons"] + available_seasons
    season_options.remove("all_seasons")
    
    selected_season = st.selectbox(
        "Choose season to analyze:",
        season_options,
        key="season_selector"
    )
    
    # Convert selection to folder name
    if selected_season == "All Seasons":
        season = "all"
        season_folder = "all_seasons"
        season_display = "All Seasons"
    else:
        season = selected_season.replace('/', '-')
        season_folder = selected_season.replace('/', '-')
        season_display = selected_season
    
    # Load data
    performance_df, totals_df, recruitment_df, message = load_competition_data(comp['folder_name'], season, season_folder)
    
    if recruitment_df is None:
        st.error(message)
        return
    
    # Display success message
    st.success(f"Loaded data for {comp['display_name']} - {season_display}")
    
    # Calculate KPIs
    total_players = len(recruitment_df)
    mean_performance = recruitment_df['performance_score'].mean()
    top_10_threshold = recruitment_df['performance_score'].quantile(0.9)
    top_10_players = len(recruitment_df[recruitment_df['performance_score'] >= top_10_threshold])
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total_players}</h2>
            <p>Total Players</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{mean_performance:.1f}</h2>
            <p>Mean Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{top_10_threshold:.1f}</h2>
            <p>Top 10% Threshold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{top_10_players}</h2>
            <p>Top 10% Players</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Descriptive Analysis Section
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-chart-bar"></i> 📈 Descriptive Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Distribution Chart
    st.plotly_chart(create_performance_distribution_chart(recruitment_df), width='stretch', key="distribution_chart")
       
    # Basic Statistics Selector
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-chart-bar"></i> 📊 Basic Statistics Leaders</h3>
    </div>
    """, unsafe_allow_html=True)
    
    stat_type = st.selectbox(
        "Select Statistics:",
        ['goals', 'xg', 'assists', 'interceptions', 'tackles', 'fouls'],
        format_func=lambda x: {
            'goals': 'Goals',
            'xg': 'Expected Goals (xG)',
            'assists': 'Assists',
            'interceptions': 'Interceptions',
            'tackles': 'Tackles',
            'fouls': 'Fouls Committed'
        }[x],
        key="basic_stats_selector"
    )
    
    st.plotly_chart(create_basic_stats_chart(totals_df, stat_type, 10), width='stretch', key=f"basic_stats_{stat_type}")
    
    # Top Performers & Specialists Section
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-trophy"></i> 🏆 Top Performers & Specialists</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        # Category selector
        category = st.selectbox(
            "Select Category:",
            ['performance_score', 'P_score', 'S_score', 'C_score', 'D_score'],
            format_func=lambda x: {
                'performance_score': 'PSCDL Performance Score',
                'P_score': 'Passing/Creation',
                'S_score': 'Finishing/Shot',
                'C_score': 'Carrying/1v1',
                'D_score': 'Defending/Disruption'
            }[x],
            key="top_specialists_selector"
        )
        
        st.plotly_chart(create_top_specialists_chart(recruitment_df, category, 10), width='stretch', key=f"top_specialists_{category}")
    
    with col2:
        # Top specialists table
        qualified_players = recruitment_df[recruitment_df['total_games'] >= 5]
        top_specialists = qualified_players.nlargest(10, category)
        
        st.subheader("Top Specialists Table")
        display_df = top_specialists[['player_name', 'team', category, 'total_games']].copy()
        display_df.columns = ['Player', 'Team', category.replace('_', ' ').title(), 'Games']
        st.dataframe(display_df, width='stretch', height=400)
    
    # Postscript
    st.info("**P.S: Only players with at least 5 games played are included.**")
    
    # Player Comparison Dashboard
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-users"></i> 👥 Player Comparison Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Player selection
    all_players = recruitment_df['player_name'].tolist()
    selected_players = st.multiselect(
        "Select Players (max 6):",
        all_players,
        default=all_players[:2] if len(all_players) >= 2 else all_players,
        max_selections=6,
        key="player_selector"
    )
    
    if selected_players:
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_pscdl_bar_chart(recruitment_df, selected_players), width='stretch', key="pscdl_bar_chart")
        
        with col2:
            st.plotly_chart(create_pscdl_radar_chart(recruitment_df, selected_players), width='stretch', key="pscdl_radar_chart")
    
    # Custom Scatter Plot Analysis
    st.markdown("""
    <div class="section-header">
        <h3><i class="fas fa-chart-scatter"></i> 📍 Custom Scatter Plot Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_axis = st.selectbox(
            "X-Axis (Horizontal):",
            ['performance_score', 'P_score', 'S_score', 'C_score', 'D_score', 'L_score', 'radar_area_normalized', 'total_games', 'total_minutes_played'],
            format_func=lambda x: {
                'performance_score': 'PSCDL Performance Score',
                'P_score': 'Passing/Creation Score',
                'S_score': 'Finishing/Shot Value Score',
                'C_score': 'Carrying/1v1 Score',
                'D_score': 'Defending/Disruption Score',
                'L_score': 'Discipline Score',
                'radar_area_normalized': 'Ability Coverage (Radar Area)',
                'total_games': 'Total Games Played',
                'total_minutes_played': 'Total Minutes Played'
            }[x],
            key="x_axis_selector"
        )
    
    with col2:
        y_axis = st.selectbox(
            "Y-Axis (Vertical):",
            ['radar_area_normalized', 'performance_score', 'P_score', 'S_score', 'C_score', 'D_score', 'L_score', 'total_games', 'total_minutes_played'],
            format_func=lambda x: {
                'performance_score': 'PSCDL Performance Score',
                'P_score': 'Passing/Creation Score',
                'S_score': 'Finishing/Shot Value Score',
                'C_score': 'Carrying/1v1 Score',
                'D_score': 'Defending/Disruption Score',
                'L_score': 'Discipline Score',
                'radar_area_normalized': 'Ability Coverage (Radar Area)',
                'total_games': 'Total Games Played',
                'total_minutes_played': 'Total Minutes Played'
            }[x],
            key="y_axis_selector"
        )
    
    st.plotly_chart(create_scatter_plot(recruitment_df, x_axis, y_axis), width='stretch', key=f"scatter_plot_{x_axis}_{y_axis}")
    
    # Analysis Tips
    st.info("""
    **Analysis Tips:**
    - **Positive Correlation:** Points trending upward indicate players who excel in both metrics
    - **Negative Correlation:** Points trending downward suggest trade-offs between skills
    - **Clusters:** Groups of players with similar performance profiles
    - **Outliers:** Players with unique skill combinations worth investigating
    """)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "index"
    
    # Route to appropriate page
    if st.session_state.current_page == "index":
        show_index_page()
    elif st.session_state.current_page == "dashboard":
        show_dashboard_page()

if __name__ == "__main__":
    main()
