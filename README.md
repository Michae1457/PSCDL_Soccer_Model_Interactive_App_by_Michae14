# ⚽ Soccer Analytics Pipeline

Soccer Analytics Dashboard - Multi-League Streamlit Version

**Author:** Michael Xu  

A Streamlit-based web application that provides an interactive dashboard for soccer analytics,
allowing users to explore player data from multiple leagues and competitions, compare performances, 
and generate insights using the PSCDL framework.

**!!!!!Further Infomation about the soccer analytics model is in my another repository in my personal profile. This repository is mainly for the interactive app!!!!!**

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

## 📄 License

This soccer analytics pipeline is provided as-is for educational and research purposes. Please ensure compliance with StatsBomb's data usage terms when using their API.

---

## 📞 Support

For questions, issues, or collaboration opportunities, please refer to the code documentation and inline comments for detailed implementation guidance.

---

*This pipeline provides a foundation for advanced soccer analytics, enabling evidence-based decision making in player recruitment, performance evaluation, and tactical planning.*
