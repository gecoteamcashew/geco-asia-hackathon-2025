
# GECO Project - Hackathon Demo

This Streamlit app demonstrates campaign ROI, platform performance, product insights, and simple forecasts.

## Setup

```bash
python3 -m venv hackathon_env
source hackathon_env/bin/activate   # Linux/Mac
hackathon_env\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Then open http://localhost:8501 in your browser.

## Data
Place your Excel/CSV files in the `data/` folder. Mock data is already included for demo purposes.
