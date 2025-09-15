# ⚽ Soccer Analytics Pipeline

A comprehensive soccer analytics pipeline for processing FA Women's Super League data from StatsBomb, transforming match-level events into player-level statistics, and providing advanced recruitment insights using the innovative PSCDL performance scoring framework.

**Author:** Michael Xu  
**Framework:** PSCDL Performance Scoring System (Passing/Creation, Finishing/Shot Value, Carrying/1v1, Defending/Disruption, Discipline)

---

## 🎯 Pipeline Overview

This pipeline provides a complete end-to-end solution for soccer analytics, from data acquisition to recruitment recommendations:

### 📋 **Pipeline Components**

1. **Data Acquisition & Setup** - StatsBomb integration for FA Women's Super League data
2. **Player Level Transformation** - Convert match events to comprehensive player statistics  
3. **Statistical Aggregation** - Create 5 structured datasets for different analysis levels
4. **Recruitment Analysis** - PSCDL framework for specialist identification and hidden gems detection
5. **Visualization & Dashboards** - Interactive tools for player comparison and analysis

### 🔬 **PSCDL Performance Scoring Framework**

The innovative 5-subscore system evaluates players across key performance dimensions:

- **P (Passing/Creation)**: Playmaking, creativity, assists (Weight: 25%)
- **S (Finishing/Shot Value)**: Clinical finishing, goal scoring (Weight: 25%)  
- **C (Carrying/1v1)**: Dribbling, ball carrying, 1v1 ability (Weight: 20%)
- **D (Defending/Disruption)**: Defensive actions, ball winning (Weight: 20%)
- **L (Discipline)**: Clean play, low foul rate (Weight: 10%)

**Formula:** `Performance Score = 0.25P + 0.25S + 0.20C + 0.20D + 0.10L`

---

## 🚀 Quick Start

### Installation

1. **Clone or download the pipeline files**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

```python
from main_pipeline import SoccerAnalyticsPipeline

# Initialize pipeline
pipeline = SoccerAnalyticsPipeline()

# Run complete pipeline
results = pipeline.run_full_pipeline()
```

### Individual Module Usage

```python
# Data scraping (Step 1)
from data_preparation import StatsBombDataScraper
scraper = StatsBombDataScraper()
scraper.scrape_all_matches("Match/Event-Level")

# Player transformation (Step 2)
from player_level_transformer import transform_to_player_level
analyzer, player_stats = transform_to_player_level(events_df)

# Statistical aggregation (Step 3)
from statistical_aggregator import StatisticalAggregator
aggregator = StatisticalAggregator()
datasets = aggregator.aggregate_player_stats("Match/Player-Level", "Statistics")

# Recruitment analysis (Step 4)
from recruitment_analyzer import RecruitmentAnalyzer
analyzer = RecruitmentAnalyzer(performance_df, totals_df)
insights = analyzer.get_recruitment_insights()

# Visualization (Step 5)
from visualization_tools import SoccerVisualizationTools
viz = SoccerVisualizationTools()
viz.create_comparison_dashboard(recruitment_df, player_names)
```

---

## 📊 Output Datasets

The pipeline generates 5 comprehensive datasets:

| Dataset | Description | Purpose |
|---------|-------------|---------|
| `statistics_totals.csv` | Raw totals (goals, assists, minutes, etc.) | Basic performance metrics |
| `statistics_per_game.csv` | Per-game averages and rates | Game-level analysis |
| `statistics_per_minute.csv` | Per-minute metrics | Efficiency analysis |
| `statistics_per_90.csv` | Standardized per-90 minute metrics | Comparative analysis |
| `statistics_performance_scores.csv` | PSCDL scores and weighted performance | Advanced analytics |

---

## 🎨 Visualization Tools

### Player Comparison Dashboard
- **Specialist Score Comparison**: Bar chart of PSCDL scores
- **Performance Profile**: Radar chart visualization  
- **Performance vs Coverage**: Scatter plot analysis
- **Overall Ability Coverage**: Radar area comparison

### Analysis Charts
- **Performance Distribution**: Overall score distribution
- **Top Performers**: Key metrics leaders
- **Specialist Analysis**: PSCDL area specialists
- **Per-90 Comparisons**: Efficiency metrics

---

## 🎯 Recruitment Insights

### Specialist Identification
- **P-Specialists**: Elite playmakers and creators
- **S-Specialists**: Clinical finishers and goal scorers
- **C-Specialists**: Dribbling and 1v1 specialists
- **D-Specialists**: Defensive stalwarts and ball winners

### Hidden Gems Detection
- High specialist potential with lower overall scores
- Undervalued market opportunities
- Development pipeline candidates

### All-Rounders Identification
- Players with high radar area coverage
- Tactical flexibility and adaptability
- Multi-position capabilities

---

## 📁 Directory Structure

```
Soccer Analytics Pipeline/
├── 00_main_pipeline.py              # Complete pipeline orchestration
├── 01_data_preparation.py           # StatsBomb data acquisition
├── 02_player_level_transformer.py   # Event-to-player transformation
├── 03_statistical_aggregator.py     # Dataset creation and aggregation
├── 04_recruitment_analyzer.py       # PSCDL analysis and insights
├── 05_visualization_tools.py        # Charts and dashboards
├── requirements.txt                 # Python dependencies
├── README.md                        # This documentation
├── Match/
│   ├── Event-Level/            # Raw match event data
│   └── Player-Level/           # Transformed player statistics
├── Statistics/                 # Aggregated statistical datasets
├── Visualizations/             # Generated charts and plots
└── Recruitment_Analysis/       # Recruitment insights and targets
```

---

## 🔧 Configuration Options

### Pipeline Parameters

```python
# Force refresh existing data
pipeline.run_full_pipeline(force_refresh=True)

# Skip saving visualization plots
pipeline.run_full_pipeline(save_plots=False)

# Custom base directory
pipeline = SoccerAnalyticsPipeline(base_dir="/path/to/your/data")
```

### PSCDL Framework Customization

```python
# Custom coefficients for performance scoring
custom_coefficients = {
    'P': 0.30,  # Increase passing weight
    'S': 0.25,  # Finishing weight
    'C': 0.20,  # Carrying weight
    'D': 0.20,  # Defending weight
    'L': 0.05   # Decrease discipline weight
}

analyzer, player_stats = transform_to_player_level(events_df, custom_coefficients)
```

---

## 📈 Key Insights & Applications

### Performance Distribution
- **Mean Performance Score**: ~50.9 (normal distribution)
- **Top 10% Threshold**: ~58.1
- **Elite Players**: <1% above 64.2

### Recruitment Value & Strategic Insights

#### 🎯 Specialist Recruitment (Immediate Impact)
- **P-Specialists**: Target for playmaking roles, creative midfield positions
- **S-Specialists**: Clinical finishers for attacking positions
- **C-Specialists**: Dribbling specialists for wide positions
- **D-Specialists**: Defensive stalwarts for backline reinforcement

#### 💎 Hidden Gems (Value Recruitment)
- **High Specialist Potential + Lower Overall Cost**: Undervalued market opportunities
- **Development Pipeline**: Young players with specialist traits
- **3-5x Return on Investment**: Development potential and squad depth

#### ⚖️ All-Rounders (Tactical Flexibility)
- **Radar Area ≥35.0**: Well-rounded players for multiple roles
- **Tactical Adaptability**: Injury cover and formation flexibility
- **Balanced Performance**: Consistent across all facets

---

## 🛠️ Advanced Features

### Radar Area Metric
- **Geometric Area Calculation**: Measures overall ability coverage
- **Theoretical Normalization**: Scales to perfect player (100)
- **Independent Assessment**: Unbiased by performance score weighting

### Interactive Comparison Tools
- **Side-by-side Player Evaluation**: Multi-player comparison dashboards
- **Specialist Score Analysis**: PSCDL framework visualization
- **Performance Profile Matching**: Tactical role identification

### Export Capabilities
- **Recruitment Targets CSV**: Categorized player recommendations
- **High-Resolution Plots**: Publication-ready visualizations
- **Structured Datasets**: Analysis-ready statistical files

---

## 🔮 Future Enhancements

### Planned Features
- **Clustering Analysis**: Player archetype identification
- **Predictive Modeling**: Performance prediction algorithms
- **Market Value Estimation**: Economic player valuation
- **Position-Specific Analysis**: Role-based performance evaluation

### Extensibility
- **Custom Metrics**: Add new performance indicators
- **Additional Data Sources**: Integrate other soccer data providers
- **Real-time Updates**: Live data processing capabilities
- **API Integration**: Web service for recruitment tools

---

## 📚 Technical Details

### Data Processing
- **Event-Level Processing**: 326 FA Women's Super League matches
- **Player Aggregation**: 395+ unique players analyzed
- **Statistical Validation**: Outlier detection and data cleaning
- **Performance Optimization**: Efficient memory usage and processing

### Quality Assurance
- **Data Validation**: Comprehensive error checking
- **Statistical Testing**: Performance score reliability
- **Visualization Standards**: Consistent chart formatting
- **Documentation**: Complete code documentation

---

## 🤝 Contributing

This pipeline is designed for extensibility and customization:

1. **Custom Metrics**: Add new performance indicators to PSCDL framework
2. **Data Sources**: Integrate additional soccer data providers
3. **Visualization**: Extend dashboard capabilities
4. **Analysis**: Implement new recruitment strategies

---

## 📄 License

This soccer analytics pipeline is provided as-is for educational and research purposes. Please ensure compliance with StatsBomb's data usage terms when using their API.

---

## 📞 Support

For questions, issues, or collaboration opportunities, please refer to the code documentation and inline comments for detailed implementation guidance.

---

*This pipeline provides a foundation for advanced soccer analytics, enabling evidence-based decision making in player recruitment, performance evaluation, and tactical planning.*
