# Streamlit Deployment Guide

## 🚀 Deploying Your Soccer Analytics Dashboard to the Cloud

This guide will help you deploy your Streamlit dashboard to various cloud platforms for public access.

## 📋 Prerequisites

1. **Install Streamlit locally first:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Test locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

## 🌐 Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push your code to GitHub:**
   - Create a new repository on GitHub
   - Upload all your files including `streamlit_app.py` and `requirements_streamlit.txt`
   - Make sure your data files are in the `Statistics/` folder

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository and branch
   - Set the main file path to `streamlit_app.py`
   - Click "Deploy!"

### Option 2: Heroku

1. **Create a Procfile:**
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy using Heroku CLI:**
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Deploy Streamlit app"
   git push heroku main
   ```

### Option 3: Railway

1. **Create a railway.json:**
   ```json
   {
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0",
       "healthcheckPath": "/"
     }
   }
   ```

2. **Deploy:**
   - Connect your GitHub repository
   - Railway will automatically detect and deploy

### Option 4: Render

1. **Create a render.yaml:**
   ```yaml
   services:
     - type: web
       name: soccer-dashboard
       env: python
       buildCommand: pip install -r requirements_streamlit.txt
       startCommand: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

## 🔧 Configuration

### Environment Variables (if needed)
Create a `.env` file or set environment variables:
```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Streamlit Configuration
Create `.streamlit/config.toml`:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#2E8B57"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F8F0"
textColor = "#262730"
```

## 📁 File Structure for Deployment

```
your-repo/
├── streamlit_app.py
├── requirements_streamlit.txt
├── 04_recruitment_analyzer.py
├── Statistics/
│   ├── statistics_performance_scores.csv
│   └── statistics_totals.csv
├── .streamlit/
│   └── config.toml
└── README.md
```

## 🚨 Important Notes

1. **Data Files:** Make sure your CSV files are included in the repository
2. **File Paths:** The app expects data files in the `Statistics/` folder
3. **Memory:** Large datasets might require paid tiers on some platforms
4. **Security:** Never commit sensitive data or API keys

## 🔍 Troubleshooting

### Common Issues:

1. **Module not found:** Ensure all dependencies are in `requirements_streamlit.txt`
2. **File not found:** Check that data files are in the correct directory
3. **Port issues:** Make sure to use `$PORT` environment variable
4. **Memory errors:** Consider using smaller datasets or paid hosting

### Debug Commands:
```bash
# Test locally
streamlit run streamlit_app.py --server.port=8501

# Check requirements
pip list

# Validate data files
python -c "import pandas as pd; print(pd.read_csv('Statistics/statistics_performance_scores.csv').shape)"
```

## 🎯 Best Practices

1. **Use caching:** `@st.cache_data` for expensive operations
2. **Error handling:** Always check if data files exist
3. **User feedback:** Show loading states and error messages
4. **Responsive design:** Test on different screen sizes
5. **Performance:** Optimize data loading and processing

## 📊 Monitoring

- **Streamlit Cloud:** Built-in analytics and usage stats
- **Heroku:** Use Heroku metrics dashboard
- **Custom:** Add Google Analytics or similar

## 🔄 Updates

To update your deployed app:
1. Push changes to your GitHub repository
2. Most platforms will automatically redeploy
3. For manual deployment, follow platform-specific instructions

---

**Need help?** Check the [Streamlit documentation](https://docs.streamlit.io) or platform-specific guides.


