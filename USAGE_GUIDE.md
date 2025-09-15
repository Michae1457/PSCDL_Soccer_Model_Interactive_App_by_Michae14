# 🚀 Soccer Analytics Pipeline - Usage Guide

## 📁 Numbered File Structure

Your pipeline now has a clear execution order with numbered files:

```
00_main_pipeline.py              # 🎯 Main orchestration script
01_data_preparation.py           # 📊 Step 1: Data scraping from StatsBomb
02_player_level_transformer.py   # 🔄 Step 2: Event-to-player transformation
03_statistical_aggregator.py     # 📈 Step 3: Statistical dataset creation
04_recruitment_analyzer.py       # 🎯 Step 4: PSCDL analysis & insights
05_visualization_tools.py        # 📊 Step 5: Charts & dashboards
```

## 🚀 Quick Start

### Option 1: Run Complete Pipeline
```bash
python3 00_main_pipeline.py
```

### Option 2: Run Individual Steps
```python
# Step 1: Data scraping
from data_preparation import StatsBombDataScraper
scraper = StatsBombDataScraper()

# Step 2: Player transformation  
from player_level_transformer import transform_to_player_level

# Step 3: Statistical aggregation
from statistical_aggregator import StatisticalAggregator

# Step 4: Recruitment analysis
from recruitment_analyzer import RecruitmentAnalyzer

# Step 5: Visualization
from visualization_tools import SoccerVisualizationTools
```

## 🔧 Import Handling

The main pipeline uses `importlib` to handle numbered module names automatically. You don't need to worry about the numbering when importing - just use the original module names!

## 📊 Pipeline Flow

```
01_data_preparation.py → 02_player_level_transformer.py → 03_statistical_aggregator.py → 04_recruitment_analyzer.py → 05_visualization_tools.py
```

## 💡 Benefits of Numbered Files

✅ **Clear Execution Order** - Easy to understand pipeline sequence  
✅ **Visual Organization** - Files appear in logical order in file explorers  
✅ **Step-by-Step Debugging** - Can run individual steps for testing  
✅ **Documentation** - Numbers serve as implicit documentation  
✅ **Maintenance** - Easy to add new steps between existing ones  

## 🎯 Usage Examples

### Run Full Pipeline
```python
from main_pipeline import SoccerAnalyticsPipeline
pipeline = SoccerAnalyticsPipeline()
results = pipeline.run_full_pipeline()
```

### Run Individual Steps
```python
# Just data scraping
from data_preparation import StatsBombDataScraper
scraper = StatsBombDataScraper()
scraper.scrape_all_matches("Match/Event-Level")

# Just player analysis
from recruitment_analyzer import RecruitmentAnalyzer
analyzer = RecruitmentAnalyzer(performance_df, totals_df)
insights = analyzer.get_recruitment_insights()
```

## 🔍 File Descriptions

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `00_main_pipeline.py` | Complete pipeline orchestration | `SoccerAnalyticsPipeline` |
| `01_data_preparation.py` | StatsBomb API integration | `StatsBombDataScraper` |
| `02_player_level_transformer.py` | Event-to-player transformation | `PlayerLevelAnalyzer`, `transform_to_player_level` |
| `03_statistical_aggregator.py` | Dataset creation & aggregation | `StatisticalAggregator` |
| `04_recruitment_analyzer.py` | PSCDL analysis & insights | `RecruitmentAnalyzer` |
| `05_visualization_tools.py` | Charts & dashboards | `SoccerVisualizationTools` |

---

*The numbered structure makes your soccer analytics pipeline easy to understand, maintain, and extend!* ⚽📊


