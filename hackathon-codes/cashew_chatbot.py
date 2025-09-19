# chatbot_app.py
# Nibbles (customer) + Clarity (staff)
# Flask chatbot with API, dynamic CSV ingestion, FAQ, and real data answers

import os, re, subprocess, sys, logging, json
from pathlib import Path
from string import Template
from typing import Dict, List
from dotenv import load_dotenv

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("chatbot")

# -----------------------------
# Dependencies
# -----------------------------
REQUIRED = ["flask", "pandas", "fuzzywuzzy", "python-levenshtein",
            "requests", "openpyxl", "python-dotenv"]
def ensure_deps():
    import importlib
    missing = []
    for pkg in REQUIRED:
        mod = "Levenshtein" if pkg == "python-levenshtein" else pkg
        try: importlib.import_module(mod)
        except ImportError: missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
ensure_deps()

import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from fuzzywuzzy import fuzz
import requests

# -----------------------------
# Load env
# -----------------------------
load_dotenv()

API_URL  = os.getenv("API_URL", "").strip()
API_KEY  = os.getenv("API_KEY", "").strip()
MODEL    = os.getenv("MODEL", "openai/gpt-oss-120b")
DATA_DIR = Path(os.getenv("DATA_DIR", r"C:\Team_Cashew_Synthetic_Data"))

ASSISTANT_NAME_CUSTOMER = "Nibbles"
ASSISTANT_NAME_STAFF    = "Clarity"

# -----------------------------
# Data Manager (dynamic CSV loader)
# -----------------------------
class DataManager:
    def __init__(self, data_dir: Path):
        self.dir = Path(data_dir)
        self.tables: Dict[str, pd.DataFrame] = {}
        self._load_all()

    def _safe_csv(self, path: Path) -> pd.DataFrame:
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception:
                df = pd.DataFrame()
        if df.empty:
            log.warning(f"[csv] Failed to load {path}")
            return df
        # cleanup
        df.columns = [str(c).replace("\u00A0", " ").strip().lower() for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
        return df

    def _load_all(self):
        log.info(f"[load] scanning {self.dir} for CSV files…")
        if not self.dir.exists():
            log.error(f"[load] directory not found: {self.dir}")
            return
        for path in self.dir.rglob("*.csv"):
            key = path.stem.lower().replace(" ", "_")
            df = self._safe_csv(path)
            self.tables[key] = df
            log.info(f"[load] {key}: {len(df)} rows, {len(df.columns)} cols")

# -----------------------------
# API caller
# -----------------------------
def call_model(messages: List[Dict[str, str]], max_tokens=512) -> str:
    if not API_KEY or not API_URL:
        return "API not configured. Please check your .env file."
    try:
        # force plain text
        messages = [{"role": "system", "content": 
                     "Always answer in plain text only. "
                     "Do not use Markdown, bold, tables, or special formatting."}] + messages

        r = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "max_tokens": max_tokens},
            timeout=20,
        )
        reply = r.json()["choices"][0]["message"]["content"].strip()

        # strip common markdown leftovers
        reply = re.sub(r"[*_`#>|]", "", reply)
        return reply
    except Exception as e:
        log.error(f"[api] error: {e}")
        return "Sorry, I couldn’t connect to the assistant API."

# -----------------------------
# Customer Assistant (FAQ + Products + API)
# -----------------------------
class CustomerQA:
    def __init__(self, dm: DataManager):
        self.dm = dm

    def _faq_lookup(self, q: str) -> str:
        faq = self.dm.tables.get("faq", pd.DataFrame())
        if faq.empty or not {"question", "answer"} <= set(faq.columns):
            return ""
        best_match = None
        best_score = 0
        for _, row in faq.iterrows():
            score = fuzz.token_set_ratio(str(row["question"]), q)
            if score > best_score:
                best_match = row["answer"]
                best_score = score
        if best_score >= 75:
            return str(best_match)
        return ""

    def _product_lookup(self, q: str) -> str:
        products = []
        for name, df in self.dm.tables.items():
            for col in df.columns:
                if re.search(r"(product|item|desc|name)", col, re.I):
                    hits = df[col][df[col].astype(str).str.contains(q, case=False, na=False)]
                    products.extend(hits.tolist())
        if products:
            uniq = sorted(set(p for p in products if isinstance(p, str)))
            return "Here’s what I found in the catalogue:\n" + "\n".join(f"- {p}" for p in uniq[:10])
        return ""

    def answer(self, q: str) -> str:
        # 1. FAQ
        faq_answer = self._faq_lookup(q)
        if faq_answer:
            return faq_answer

        # 2. Products
        prod_answer = self._product_lookup(q)
        if prod_answer:
            return prod_answer

        # 3. Fallback → API with CSO persona and sample data
        faq_df = self.dm.tables.get("faq", pd.DataFrame()).head(10)
        prod_df = self.dm.tables.get("product_inventory", pd.DataFrame()).head(10)

        faq_examples = faq_df.to_dict(orient="records") if not faq_df.empty else []
        prod_examples = prod_df.to_dict(orient="records") if not prod_df.empty else []

        return call_model([
            {"role": "system", "content": (
                "You are Nibbles, a warm and professional Customer Service Officer at Camel Nuts. "
                "Base your answers on the company FAQ and product catalogue if possible. "
                "If nothing is found, fall back to general knowledge but keep answers short, "
                "accurate, and in a natural customer service tone. "
                "Never mention that you are an AI. Always respond as a real Camel Nuts staff member. "
                f"Here are some sample FAQs: {json.dumps(faq_examples, ensure_ascii=False)} "
                f"Here are some sample products: {json.dumps(prod_examples, ensure_ascii=False)}"
            )},
            {"role": "user", "content": q},
        ])

# -----------------------------
# Staff Assistant (Sales analytics + API)
# -----------------------------
class StaffAnalytics:
    def __init__(self, dm: DataManager):
        self.dm = dm

    def sales_summary(self, query: str) -> str:
        df = None
        for k, v in self.dm.tables.items():
            if "sales" in k and not v.empty:
                df = v
                break
        if df is None:
            return "No sales data found."

        # detect revenue column
        rev_col = next((c for c in df.columns if re.search(r"(amount|revenue|sales)", c, re.I)), None)
        if not rev_col:
            return "Sales file has no revenue/amount column."

        # detect product column
        prod_col = next((c for c in df.columns if re.search(r"(desc|product|item|name)", c, re.I)), None)
        if not prod_col:
            return "Sales file has no product description column."

        # detect date column
        date_col = next((c for c in df.columns if re.search(r"(date|time)", c, re.I)), None)

        df = df.copy()
        df[rev_col] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0)

        # filter by year if present
        m = re.search(r"(20\d{2})", query)
        if date_col and m:
            year = int(m.group(1))
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[df[date_col].dt.year == year]

        grouped = df.groupby(prod_col)[rev_col].sum().sort_values(ascending=False)

        # detect top/bottom N
        m_top = re.search(r"top\s+(\d+)", query, re.I)
        m_bot = re.search(r"bottom\s+(\d+)", query, re.I)
        if m_top:
            n = int(m_top.group(1))
            head = grouped.head(n)
            return "\n".join([f"Top {n} products by revenue:"] + [f"- {p}: ${v:,.2f}" for p, v in head.items()])
        if m_bot:
            n = int(m_bot.group(1))
            tail = grouped.tail(n)
            return "\n".join([f"Bottom {n} products by revenue:"] + [f"- {p}: ${v:,.2f}" for p, v in tail.items()])

        return f"Total revenue: ${grouped.sum():,.2f}"

    def answer(self, q: str) -> str:
        if "sale" in q.lower() or "revenue" in q.lower():
            return self.sales_summary(q)
        return call_model([
            {"role": "system", "content": (
                "You are Clarity, a Camel Nuts internal staff analytics officer. "
                "Use the CSV data for analysis if possible. "
                "If the query is outside the data scope, answer helpfully but concisely. "
                "Do not use markdown or formatting."
            )},
            {"role": "user", "content": q},
        ])

# -----------------------------
# Flask App + Full HTML Template
# -----------------------------
app = Flask(__name__)

INDEX_HTML_TPL = Template("""
<!doctype html>
<html>
<head>
  <title>${CUSTOMER}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
  :root{ --gold:#d2b48c; --gold-dark:#c19a6b; --bg:#f7f5f2; }
  *{box-sizing:border-box}
  body{background:var(--bg);font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0}
  .container{max-width:720px;margin:0 auto;padding:20px}
  .widget{background:#fff;border-radius:18px;box-shadow:0 8px 30px rgba(0,0,0,.08);overflow:hidden}
  .header{background:linear-gradient(135deg,var(--gold),#e9c77d);color:#2b2b2b;padding:20px;text-align:center}
  .header h1{margin:0 0 6px;font-size:22px}
  .header p{margin:0;opacity:.85;font-size:13px}
  .chat{height:520px;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px;background:#fbfaf7}
  .bubble{padding:12px 14px;border-radius:16px;max-width:85%;line-height:1.4;white-space:pre-wrap;position:relative}
  .user{align-self:flex-end;background:var(--gold);color:#1f160d}
  .bot{align-self:flex-start;background:#ffffff;color:#2b2b2b;border:1px solid #eee}
  .badge{font-size:11px;font-weight:600;opacity:.8;margin-bottom:4px}
  .composer{display:flex;padding:14px;border-top:1px solid #eee;background:#fff}
  .input-wrap{flex:1;position:relative}
  textarea{width:100%;border-radius:24px;padding:12px 88px 12px 14px;border:2px solid #eee;resize:none;font-family:inherit;font-size:14px;outline:none}
  textarea:focus{border-color:var(--gold)}
  button{position:absolute;right:8px;top:50%;transform:translateY(-50%);background:var(--gold);color:#1f160d;border:none;padding:8px 16px;border-radius:18px;cursor:pointer;font-size:14px}
  button:hover{background:var(--gold-dark)}
  .typing{color:#666;font-style:italic;align-self:flex-start}
  .hint{font-size:12px;color:#6b6b6b;margin-top:6px}
  </style>
</head>
<body>
  <div class="container">
    <div class="widget">
      <div class="header">
        <h1>${CUSTOMER}</h1>
        <p>${CUSTOMER} serves customers from FAQ + CSV data. ${STAFF} answers staff analytics. Prefix with “Staff query:”</p>
      </div>
      <div id="chat" class="chat">
        <div class="bubble bot">
          <div class="badge">${CUSTOMER}</div>
Hello! I'm ${CUSTOMER}. I can help you with:
• Product information and availability (from CSVs)
• Orders and shipping (FAQ-based)
• Nutritional info and allergens (FAQ or catalogue-based)
• For staff analytics, start your message with “Staff query:”
        </div>
        <div class="hint">Tip: You can chat in your language. Staff analytics still need the “Staff query:” prefix.</div>
      </div>
      <div class="composer">
        <div class="input-wrap">
          <textarea id="msg" rows="1" placeholder="Type your message…" onkeydown="handleEnter(event)"></textarea>
          <button onclick="sendMsg()">Send</button>
        </div>
      </div>
    </div>
  </div>

<script>
function handleEnter(e){ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMsg(); } }
function autoResize(el){ el.style.height='auto'; el.style.height=el.scrollHeight + 'px'; }
document.getElementById('msg').addEventListener('input', function(){ autoResize(this); });

function escapeHtml(str){
  return str.replace(/[&<>'"]/g, function(tag){
    const chars = { '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' };
    return chars[tag] || tag;
  });
}

async function sendMsg(){
  const box=document.getElementById('msg');
  const text=box.value.trim();
  if(!text) return;
  const chat=document.getElementById('chat');
  chat.insertAdjacentHTML('beforeend','<div class="bubble user">'+escapeHtml(text)+'</div>');
  box.value=''; autoResize(box); chat.scrollTop=chat.scrollHeight;

  const typing=document.createElement('div');
  typing.className='typing'; typing.innerText='Assistant is typing…';
  chat.appendChild(typing); chat.scrollTop=chat.scrollHeight;

  let resp = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:text})});
  let data = await resp.json();
  chat.removeChild(typing);

  const speaker = escapeHtml(data.speaker || '${CUSTOMER}');
  const reply   = escapeHtml(data.reply || '');
  chat.insertAdjacentHTML('beforeend','<div class="bubble bot"><div class="badge">'+speaker+'</div>'+reply+'</div>');
  chat.scrollTop=chat.scrollHeight;
}
</script>
</body>
</html>
""")

dm = DataManager(DATA_DIR)
cust = CustomerQA(dm)
staff = StaffAnalytics(dm)

@app.route("/")
def index():
    return render_template_string(INDEX_HTML_TPL.safe_substitute(
        CUSTOMER=ASSISTANT_NAME_CUSTOMER, STAFF=ASSISTANT_NAME_STAFF))

@app.route("/chat", methods=["POST"])
def chat():
    msg = (request.json.get("message") or "").strip()
    if not msg:
        return jsonify({"reply": "Say something!"})

    if msg.lower().startswith("staff query:"):
        reply = staff.answer(msg.split(":",1)[1].strip())
        speaker = ASSISTANT_NAME_STAFF
    else:
        reply = cust.answer(msg)
        speaker = ASSISTANT_NAME_CUSTOMER

    return jsonify({"reply": reply, "speaker": speaker})

if __name__ == "__main__":
    print("🚀 Starting server on http://127.0.0.1:5000/")
    app.run(host="127.0.0.1", port=5000)
