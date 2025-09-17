import os
import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

# --- Auto-install dependencies ---
REQUIRED = ["flask", "requests", "pandas", "python-dateutil", "waitress"]
def ensure_deps():
    import importlib
    missing = []
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

ensure_deps()

from flask import Flask, request, jsonify, render_template_string
import requests
import pandas as pd
from dateutil import parser as dateparser

# --- Config ---
BITDEER_API_KEY = os.getenv("BITDEER_API_KEY", "").strip()
BITDEER_CHAT_URL = "https://api-inference.bitdeer.ai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

DATA_DIR = Path(r"C:\Team_Cashew_Synthetic_Data")
SALES_CSV = DATA_DIR / "sales_data.csv"
SALES_CSV_FALLBACK = DATA_DIR / "product_platform_sales.csv"

# --- App ---
app = Flask(__name__)

# --- Helpers ---
def safe_read_csv(path: Path):
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return pd.DataFrame()

def norm_cols(df):
    if df is None or df.empty: return df
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def to_datetime_safe(val):
    try:
        return dateparser.parse(str(val))
    except Exception:
        return None

def sanitize_float(x):
    try: return float(x)
    except: return None

def parse_date_range_from_text(text):
    now = datetime.now()
    t = (text or "").lower()
    if "last month" in t:
        first_this = now.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        last_month_end = first_this - timedelta(seconds=1)
        first_last = last_month_end.replace(day=1)
        return first_last, last_month_end
    if "this month" in t:
        first_this = now.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
        return first_this, now
    if "this year" in t:
        first = now.replace(month=1,day=1,hour=0,minute=0,second=0,microsecond=0)
        return first, now
    m = re.search(r"last\s+(\d+)\s*days?", t)
    if m:
        n=int(m.group(1))
        return now - timedelta(days=n), now
    return None,None

# --- Load Data ---
sales_df = norm_cols(safe_read_csv(SALES_CSV)) if SALES_CSV.exists() else norm_cols(safe_read_csv(SALES_CSV_FALLBACK))
SALE_DATE_COL = next((c for c in sales_df.columns if "posting" in c or "date" in c), None)
SALE_NAME_COL = next((c for c in sales_df.columns if "product" in c or "name" in c), None)
SALE_UNITS_COL = next((c for c in sales_df.columns if "unit" in c or "qty" in c), None)
SALE_REV_COL = next((c for c in sales_df.columns if "rev" in c or "sales" in c or "amount" in c), None)

# --- Sales Query ---
def sales_by_products(query):
    if sales_df.empty or not SALE_NAME_COL: return None
    q = (query or "").lower()
    start,end = parse_date_range_from_text(q)
    df = sales_df.copy()
    if SALE_DATE_COL:
        df["_date"]=df[SALE_DATE_COL].apply(to_datetime_safe)
        if start: df=df[df["_date"]>=start]
        if end: df=df[df["_date"]<=end]
    products = []
    for name in df[SALE_NAME_COL].dropna().unique():
        n=name.lower()
        if any(w in q or n in q for w in q.split() if len(w)>3) or n in q:
            products.append(name)
    if not products: return None
    results=[]
    for p in products:
        d=df[df[SALE_NAME_COL].astype(str).str.lower().str.contains(p.lower())]
        if d.empty: continue
        units=d[SALE_UNITS_COL].apply(sanitize_float).sum() if SALE_UNITS_COL else 0
        rev=d[SALE_REV_COL].apply(sanitize_float).sum() if SALE_REV_COL else 0
        results.append({"product":p,"units":int(units),"revenue":round(rev,2)})
    return results if results else None

# --- LLM helpers ---
def build_system_prompt():
    return "You are a friendly customer support assistant. Use provided sales data context when available. Speak naturally like a human rep."

def build_user_prompt(user_msg, sales_info):
    parts=[f"User message:\n{user_msg}\n"]
    if sales_info:
        parts.append("Sales data summary:")
        for r in sales_info:
            parts.append(f"- {r['product']}: {r['units']} units, ${r['revenue']} revenue (Posting_Date based)")
    return "\n".join(parts)

def call_bitdeer_chat(messages):
    headers={"Authorization":f"Bearer {BITDEER_API_KEY}","Content-Type":"application/json"}
    payload={"model":MODEL,"messages":messages,"max_tokens":400,"temperature":0.4}
    r=requests.post(BITDEER_CHAT_URL,headers=headers,data=json.dumps(payload))
    return r.json().get("choices",[{}])[0].get("message",{}).get("content","")

# --- UI ---
INDEX_HTML = """
<!doctype html>
<html><head>
<title>Elephant Snacks Chat</title>
<style>
body{background:#f5f5f5;font-family:sans-serif}
.widget{max-width:420px;margin:20px auto;border:1px solid #ccc;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.1);overflow:hidden}
.header{background:#d2b48c;color:white;padding:14px}
.chat{height:420px;overflow-y:auto;padding:10px;display:flex;flex-direction:column;gap:8px;background:white}
.bubble{padding:10px 14px;border-radius:16px;max-width:80%}
.user{align-self:flex-end;background:#d2b48c;color:#000}
.bot{align-self:flex-start;background:#fff;border:1px solid #ddd}
.composer{display:flex;padding:10px;border-top:1px solid #ddd;background:#fafafa}
textarea{flex:1;border-radius:20px;padding:8px;border:1px solid #ccc;resize:none}
button{margin-left:8px;background:#d2b48c;color:white;border:none;padding:8px 14px;border-radius:20px;cursor:pointer}
</style></head>
<body>
<div class="widget">
 <div class="header"><b>Hi there 👋</b><br><small>Our operating hours are 8am–5pm (SGT)</small></div>
 <div id="chat" class="chat"></div>
 <div class="composer">
   <textarea id="msg" rows="1" placeholder="Enter your message..."></textarea>
   <button onclick="sendMsg()">Send</button>
 </div>
</div>
<script>
async function sendMsg(){
  const box=document.getElementById('msg');const text=box.value.trim();
  if(!text)return;const chat=document.getElementById('chat');
  chat.innerHTML+="<div class='bubble user'>"+text+"</div>";box.value='';
  const res=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text})});
  const data=await res.json();
  chat.innerHTML+="<div class='bubble bot'>"+data.reply+"</div>";chat.scrollTop=chat.scrollHeight;
}
</script></body></html>
"""

@app.route("/")
def index(): return render_template_string(INDEX_HTML)

@app.route("/chat",methods=["POST"])
def chat_route():
    data=request.get_json(force=True); user_msg=data.get("message","")
    sales_info=sales_by_products(user_msg)
    sys_prompt=build_system_prompt(); user_prompt=build_user_prompt(user_msg,sales_info)
    messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}]
    reply=call_bitdeer_chat(messages)
    return jsonify({"reply":reply})

def run():
    try:
        from waitress import serve
        serve(app,host="127.0.0.1",port=5000)
    except: app.run(port=5000)

if __name__=="__main__": run()
