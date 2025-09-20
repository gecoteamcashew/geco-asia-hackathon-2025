# app.py
#   Cashew4Life - Dashboard (Latest Patched Version 0.1f)
# - Hide Diagnostics page
# - Remove empty Inventory Snapshot
# - Rename NLQ page
# - Fix month/year parsing for queries (e.g. "revenue shopee pistachios oct 2024")
# - Add support for "top five product sales"
# - Handle irrelevant queries gracefully

from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Union

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from contextlib import contextmanager
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ================== Config ==================
APP_TITLE = "Cashew4Life - Dashboard"
DATA_DIR = Path("data")
DB_PATH = Path("hackathon.sqlite")
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------- THEME FIX: Larger, Bolder Fonts ----------
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
  font-size: 1.2rem;     /* bigger base font */
  color: #222222;        /* darker text */
}
h1, h2, h3, h4 {
  font-weight: 800;      /* bolder headings */
}
h1 { font-size: 2.1rem; }
h2 { font-size: 1.7rem; }
h3 { font-size: 1.4rem; }
h4 { font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# ================== Helpers ==================
def norm_col(c: str) -> str:
    c = c.strip()
    c = c.replace("(Clean)", "clean")
    c = c.replace(" ", "_")
    c = re.sub(r"[^A-Za-z0-9_]+", "_", c)
    return c.lower().strip("_")

def to_series(x, index=None) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x, index=index)
    return pd.Series([x])

def strip_non_numeric(x) -> pd.Series:
    s = to_series(x)
    if s.dtype == object:
        s = s.replace(r"[^0-9.\-]", "", regex=True)
    return s

def coerce_float(x, default: float = 0.0) -> pd.Series:
    s = strip_non_numeric(x)
    s = pd.to_numeric(s, errors="coerce").fillna(default).astype(float)
    return s

def coerce_int(x, default: int = 0) -> pd.Series:
    s = strip_non_numeric(x)
    s = pd.to_numeric(s, errors="coerce").fillna(default).astype(int)
    return s

def coerce_date(x, fallback: Optional[pd.Timestamp] = None) -> pd.Series:
    s = to_series(x)
    out = pd.to_datetime(s, errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if fallback is not None:
        out = out.fillna(fallback)
    return out

def norm_code_key(x: str) -> str:
    return str(x).strip().upper()

@contextmanager
def sqlite_conn(path: Path):
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()

def write_df(conn: sqlite3.Connection, df: pd.DataFrame, table: str):
    if df is not None and not df.empty:
        df.to_sql(table, conn, if_exists="replace", index=False)

def read_csv_any(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [norm_col(c) for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Failed to read {path.name}: {e}")
        return pd.DataFrame()

# ================== File map ==================
@dataclass
class FileMap:
    ecommerce_purchases: Optional[Path] = None
    sales_transactions: Optional[Path] = None
    sku_master: Optional[Path] = None
    traffic_acquisition: Optional[Path] = None  # for Campaign ROI

def get_filemap(data_dir: Path) -> FileMap:
    fm = FileMap()
    files = {p.name.lower(): p for p in data_dir.glob("*.csv")}
    fm.ecommerce_purchases = files.get("ecommerce_purchases.csv")
    fm.sales_transactions = files.get("sales_transactions.csv")
    fm.sku_master = files.get("sku_master.csv")
    fm.traffic_acquisition = (
        files.get("traffic_acquisition.csv")
        or files.get("marketing.csv")
        or files.get("marketing_performance.csv")
        or files.get("channels.csv")
        or files.get("campaigns.csv")
    )
    return fm

# ================== Normalizers ==================
# Ecommerce_Purchases.csv
ECOM_ALIAS = {
    "order_date": "date",
    "item_name": "sku",
    "quantity": "quantity",
    "unit_price": "unit_price",
    "platform": "platform",
    "order_no": "order_no",
    "customer_id": "customer_id",
    "payment_type": "payment_type",
}

def normalize_ecommerce_purchases(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()

    for src, tgt in ECOM_ALIAS.items():
        if src in out.columns and tgt not in out.columns:
            out[tgt] = out[src]

    out["date"] = coerce_date(out.get("date", pd.Series(pd.NaT)), fallback=pd.Timestamp("2023-01-01"))

    if "sku" not in out.columns:
        text_cols = [c for c in out.columns if out[c].dtype == object]
        out["sku"] = out[text_cols[0]] if text_cols else "UNKNOWN"

    out["quantity"] = coerce_int(out.get("quantity", 0)).astype(int)
    out["unit_price"] = coerce_float(out.get("unit_price", 0.0)).astype(float)

    out["discount_pct"] = 0.0
    out["platform"] = out.get("platform", "Unknown").fillna("Unknown")

    out["item_revenue"] = (out["quantity"] * out["unit_price"]).astype(float)
    out["gross_sales"] = out["item_revenue"]
    out["discounts"] = (
        pd.Series(out["gross_sales"]) * pd.Series(out["discount_pct"]).astype(float)
    ).astype(float)

    rr = out.get("returns_refunds", None)
    out["returns_refunds"] = (
        coerce_float(rr, 0.0).astype(float) if rr is not None else 0.0
    )

    out["net_sales"] = (
        out["gross_sales"] - out["discounts"] - out["returns_refunds"]
    ).astype(float)
    out["campaign"] = out.get("campaign", "Unknown")

    keep = [
        "date","sku","quantity","unit_price","discount_pct","item_revenue","platform",
        "campaign","returns_refunds","gross_sales","discounts","net_sales",
        "order_no","customer_id","payment_type"
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    return out


# Sales_Transactions.csv
TX_ALIAS = {
    "posting_date": "date", "order_date": "date", "transaction_date": "date", "date": "date",
    "item_code": "sku", "description": "sku", "item_name": "sku", "product_sku": "sku", "item": "sku",
    "quantity_clean": "quantity", "quantity": "quantity", "qty": "quantity",
    "unit_price": "unit_price", "price": "unit_price",
    "line_amount": "item_revenue", "net_amount_excl_vat": "item_revenue",
    "total_amount_excl_vat": "net_sales",
    "line_discount_percent": "discount_pct", "discount_percent": "discount_pct",
    "sales_channel": "platform", "platform": "platform",
    "promo_flag": "promo_flag", "returns_refunds": "returns_refunds",
    "campaign": "campaign",
}

def normalize_sales_transactions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()

    for src, tgt in TX_ALIAS.items():
        if src in out.columns and tgt not in out.columns:
            out[tgt] = out[src]

    if "date" not in out.columns:
        for c in out.columns:
            if "date" in c:
                out["date"] = out[c]
                break
    out["date"] = coerce_date(
        out.get("date", pd.Series(pd.NaT)), fallback=pd.Timestamp("2023-01-01")
    )

    if "sku" not in out.columns:
        text_cols = [c for c in out.columns if out[c].dtype == object]
        out["sku"] = out[text_cols[0]] if text_cols else "UNKNOWN"

    out["quantity"] = coerce_int(out.get("quantity", 0)).astype(int)
    out["unit_price"] = coerce_float(out.get("unit_price", 0.0)).astype(float)
    out["discount_pct"] = coerce_float(out.get("discount_pct", 0.0)).clip(0.0, 0.95).astype(float)

    if "item_revenue" in out.columns:
        out["item_revenue"] = coerce_float(out["item_revenue"], 0.0).astype(float)
    else:
        out["item_revenue"] = (
            out["quantity"] * out["unit_price"] * (1 - out["discount_pct"]).astype(float)
        ).astype(float)

    out["platform"] = out.get("platform", "Unknown").fillna("Unknown")
    out["campaign"] = out.get("campaign", "Unknown")

    rr = out.get("returns_refunds", None)
    out["returns_refunds"] = (
        coerce_float(rr, 0.0).astype(float) if rr is not None else 0.0
    )

    out["gross_sales"] = (out["quantity"] * out["unit_price"]).astype(float)
    out["discounts"] = (out["gross_sales"] * out["discount_pct"]).astype(float)

    if "net_sales" in out.columns:
        ns = coerce_float(out["net_sales"], 0.0)
        if float(ns.sum()) == 0.0 and float(out["item_revenue"].sum()) > 0.0:
            out["net_sales"] = (
                out["gross_sales"] - out["discounts"] - out["returns_refunds"]
            ).astype(float)
        else:
            out["net_sales"] = ns.astype(float)
    else:
        out["net_sales"] = (
            out["gross_sales"] - out["discounts"] - out["returns_refunds"]
        ).astype(float)

    keep = [
        "date","sku","quantity","unit_price","discount_pct","item_revenue","platform",
        "campaign","returns_refunds","gross_sales","discounts","net_sales"
    ]
    out = out[[c for c in keep if c in out.columns]].copy()
    return out


# SKU master: cogs, inventory, robust description map
def normalize_sku_master(df: pd.DataFrame) -> Dict[str, Union[pd.DataFrame, Dict[str, str]]]:
    out: Dict[str, Union[pd.DataFrame, Dict[str, str]]] = {
        "cogs": pd.DataFrame(), "inventory": pd.DataFrame(), "sku_desc_map": {}
    }
    if df.empty:
        return out

    work = df.copy()
    if "sku" not in work.columns:
        for c in work.columns:
            if c in ["item_code", "item", "item_name", "product_sku", "description"]:
                work["sku"] = work[c]
                break
    if "sku" not in work.columns:
        return out

    sku_desc_map: Dict[str, str] = {}
    desc_col = "description" if "description" in work.columns else None
    code_col = "item_code" if "item_code" in work.columns else None

    if desc_col:
        if code_col:
            for code, desc in zip(work[code_col].astype(str), work[desc_col].astype(str)):
                sku_desc_map[code] = desc
                sku_desc_map[norm_code_key(code)] = desc
        for s, desc in zip(work["sku"].astype(str), work[desc_col].astype(str)):
            sku_desc_map[s] = desc
            sku_desc_map[norm_code_key(s)] = desc

    cogs = pd.DataFrame({"sku": work["sku"]})
    cost_like = None
    for c in ["cogs_per_unit", "unit_cost", "avg_cost", "cost"]:
        if c in work.columns:
            cost_like = c
            break
    if cost_like is None:
        alt = [c for c in work.columns if ("cog" in c or "cost" in c)]
        if alt:
            cost_like = alt[0]
    cogs["cogs_per_unit"] = (
        coerce_float(work[cost_like], 0.0).astype(float) if cost_like else 0.0
    )
    cogs = cogs.dropna(subset=["sku"]).drop_duplicates()

    inv = pd.DataFrame({"sku": work["sku"]})
    has_soh = "stock_on_hand" in work.columns
    has_ltd = "lead_time_days" in work.columns
    if has_soh:
        inv["stock_on_hand"] = coerce_int(work["stock_on_hand"], 0).astype(int)
    if has_ltd:
        inv["lead_time_days"] = coerce_int(work["lead_time_days"], 14).astype(int)
    if has_soh or has_ltd:
        if not has_soh:
            inv["stock_on_hand"] = 0
        if not has_ltd:
            inv["lead_time_days"] = 14
        inv = inv.dropna(subset=["sku"]).drop_duplicates()
    else:
        inv = pd.DataFrame()

    out["cogs"] = cogs
    out["inventory"] = inv
    out["sku_desc_map"] = sku_desc_map
    return out


# ================== Loader ==================
@st.cache_data(show_spinner=False)
def load_all() -> Dict[str, Union[pd.DataFrame, Dict]]:
    fm = get_filemap(DATA_DIR)

    raw_ep = read_csv_any(fm.ecommerce_purchases)
    raw_st = read_csv_any(fm.sales_transactions)

    norm_ep = normalize_ecommerce_purchases(raw_ep)
    norm_st = normalize_sales_transactions(raw_st)

    purchases = pd.concat(
        [df for df in [norm_ep, norm_st] if not df.empty],
        ignore_index=True
    )
    if not purchases.empty:
        purchases = purchases.dropna(subset=["date","sku"]).reset_index(drop=True)

    raw_sku = read_csv_any(fm.sku_master)
    parts = normalize_sku_master(raw_sku)
    cogs = parts["cogs"]
    inventory = parts["inventory"]
    sku_desc_map: Dict[str, str] = parts.get("sku_desc_map", {})

    if not purchases.empty:
        if not cogs.empty:
            purchases = purchases.merge(cogs, on="sku", how="left")
        else:
            purchases["cogs_per_unit"] = 0.0
        purchases["cogs_per_unit"] = purchases["cogs_per_unit"].fillna(0.0).astype(float)
        purchases["contribution_margin"] = (
            purchases["item_revenue"] - purchases["cogs_per_unit"] * purchases["quantity"]
        ).astype(float)

        # Primary map: SKU_Master (raw + normalized)
        purchases["SKU_Description"] = purchases["sku"].astype(str).map(sku_desc_map)
        purchases.loc[purchases["SKU_Description"].isna(), "SKU_Description"] = \
            purchases.loc[purchases["SKU_Description"].isna(), "sku"].astype(str).map(
                lambda s: sku_desc_map.get(norm_code_key(s))
            )

        # Fallback: infer Description from Sales_Transactions.description if available
        if not raw_st.empty and "description" in raw_st.columns:
            st_map: Dict[str, str] = {}
            code_col = None
            for cand in ["item_code", "item", "sku", "description"]:
                if cand in raw_st.columns:
                    code_col = cand
                    break
            if code_col is not None:
                for code, desc in zip(raw_st[code_col].astype(str), raw_st["description"].astype(str)):
                    st_map[code] = desc
                    st_map[norm_code_key(code)] = desc
                mask = purchases["SKU_Description"].isna()
                if mask.any():
                    purchases.loc[mask, "SKU_Description"] = purchases.loc[mask, "sku"].astype(str).map(st_map)
                    purchases.loc[mask & purchases["SKU_Description"].isna(), "SKU_Description"] = \
                        purchases.loc[mask, "sku"].astype(str).map(lambda s: st_map.get(norm_code_key(s)))

        # Last resort: use raw sku
        purchases["SKU_Description"] = purchases["SKU_Description"].fillna(purchases["sku"].astype(str))

    with sqlite_conn(DB_PATH) as conn:
        write_df(conn, purchases, "purchases")
        write_df(conn, inventory, "inventory")
        write_df(conn, cogs, "cogs")

    traffic = read_csv_any(fm.traffic_acquisition)
    if not traffic.empty:
        ren = {}
        if "date" not in traffic.columns:
            for c in traffic.columns:
                if c in ["event_date", "day", "ds"]:
                    ren[c] = "date"
                    break
        if "session_primary_channel_group" in traffic.columns:
            ren["session_primary_channel_group"] = "channel"
        elif "event_type" in traffic.columns:
            ren["event_type"] = "channel"
        if "marketing_spend" in traffic.columns:
            ren["marketing_spend"] = "spend"
        if "attributed_sales" in traffic.columns:
            ren["attributed_sales"] = "revenue_attributed"

        traffic = traffic.rename(columns=ren)
        traffic["date"] = coerce_date(
            traffic.get("date", pd.Series(pd.NaT)),
            fallback=pd.Timestamp("2023-01-01")
        )
        if "channel" not in traffic.columns:
            traffic["channel"] = "Unknown"
        if "spend" not in traffic.columns:
            traffic["spend"] = 0.0
        if "revenue_attributed" not in traffic.columns:
            traffic["revenue_attributed"] = 0.0

    return {
        "files": {
            "Ecommerce_Purchases": str(fm.ecommerce_purchases) if fm.ecommerce_purchases else None,
            "Sales_Transactions": str(fm.sales_transactions) if fm.sales_transactions else None,
            "SKU_Master": str(fm.sku_master) if fm.sku_master else None,
            "Traffic_Acquisition": str(fm.traffic_acquisition) if fm.traffic_acquisition else None,
        },
        "raw_ep": raw_ep,
        "raw_st": raw_st,
        "purchases": purchases,
        "inventory": inventory,
        "cogs": cogs,
        "sku_desc_map": sku_desc_map,
        "traffic": traffic,
    }


# ================== Forecast helper ==================
def forecast_overall_quantity_monthly(purchases: pd.DataFrame, periods=6) -> pd.DataFrame:
    if purchases.empty:
        return pd.DataFrame()
    s = purchases[["date","quantity"]].copy()
    s["date"] = coerce_date(s["date"])
    s = s.set_index("date").resample("MS")["quantity"].sum().fillna(0.0)
    if len(s) < 3:
        return pd.DataFrame()
    try:
        model = ExponentialSmoothing(
            s, seasonal_periods=12, trend="add", seasonal="add", initialization_method="estimated"
        )
        fit = model.fit()
        fc = fit.forecast(periods)
        return pd.DataFrame({"month": fc.index, "forecast_quantity": fc.values})
    except Exception as e:
        st.warning(f"Forecast error: {e}")
        return pd.DataFrame()
# ================== UI Pages ==================
def header():
    st.markdown(f"<h2 style='margin:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown("---")


def page_home(d: Dict[str, pd.DataFrame]):
    purchases = d["purchases"]
    if purchases.empty or (
        float(purchases["quantity"].sum()) == 0.0
        and float(purchases["net_sales"].sum()) == 0.0
        and float(purchases["item_revenue"].sum()) == 0.0
    ):
        st.error("Purchases still zero. Please check your input data files.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Net Sales", f"${purchases['net_sales'].sum():,.0f}")
    c2.metric("Total Units Sold", f"{int(purchases['quantity'].sum()):,}")
    c3.metric(
        "Contribution Margin",
        f"${purchases.get('contribution_margin', pd.Series(dtype=float)).sum():,.0f}",
    )
    c4.metric("Rows", f"{len(purchases):,}")

    st.markdown("---")
    st.subheader("Top SKUs by Sales")
    by_sku = (
        purchases.groupby("SKU_Description", dropna=False)["net_sales"]
        .sum()
        .reset_index()
        .sort_values("net_sales", ascending=False)
    )
    by_sku_disp = by_sku.rename(columns={"net_sales": "sales"})
    fig = px.bar(by_sku_disp.head(10), x="SKU_Description", y="sales")
    fig.update_layout(xaxis_title="SKU (Description)")
    st.plotly_chart(fig, use_container_width=True)


def page_platform(d: Dict[str, pd.DataFrame]):
    p = d["purchases"]
    if p.empty:
        st.warning("No purchases data.")
        return
    plat = (
        p.groupby("platform", dropna=False)
        .agg(
            gross_sales=("gross_sales", "sum"),
            discounts=("discounts", "sum"),
            returns=("returns_refunds", "sum"),
            net_sales=("net_sales", "sum"),
            quantity=("quantity", "sum"),
        )
        .reset_index()
        .sort_values("net_sales", ascending=False)
    )
    st.dataframe(plat, use_container_width=True)
    if not plat.empty:
        fig = px.bar(
            plat,
            x="platform",
            y=["gross_sales", "discounts", "returns", "net_sales"],
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)


def page_products(d: Dict[str, pd.DataFrame]):
    p = d["purchases"]
    inv = d["inventory"]
    if p.empty:
        st.warning("No purchases data.")
        return

    prod = (
        p.groupby("SKU_Description", dropna=False)
        .agg(
            total_net_sales=("net_sales", "sum"),
            total_units=("quantity", "sum"),
            contribution_margin=("contribution_margin", "sum"),
        )
        .reset_index()
        .sort_values("total_net_sales", ascending=False)
    )
    prod_disp = prod.rename(columns={"total_net_sales": "total_sales"})
    st.dataframe(
        prod_disp.rename(columns={"SKU_Description": "SKU"}), use_container_width=True
    )

    if not prod.empty:
        fig = px.bar(
            prod.sort_values("contribution_margin", ascending=False).head(12),
            x="SKU_Description",
            y="contribution_margin",
        )
        fig.update_layout(xaxis_title="SKU (Description)")
        st.plotly_chart(fig, use_container_width=True)

    # Inventory Snapshot only if we have data
    if not inv.empty:
        st.markdown("---")
        st.subheader("Inventory Snapshot")
        inv_show = inv.copy().rename(columns={"sku": "SKU"})
        st.dataframe(inv_show, use_container_width=True)


def page_forecasts(d: Dict[str, pd.DataFrame]):
    p = d["purchases"]
    if p.empty:
        st.warning("No purchases data.")
        return

    skus = ["Overall"] + sorted(p["SKU_Description"].dropna().unique().tolist())
    sel = st.selectbox("Forecast for", skus, index=0)

    df = p.copy()
    df["date"] = coerce_date(df["date"])
    if sel != "Overall":
        df = df[df["SKU_Description"] == sel]

    monthly = (
        df.set_index("date")
        .resample("MS")["quantity"]
        .sum()
        .rename("quantity")
        .reset_index()
    )

    if monthly["quantity"].sum() == 0 or len(monthly) < 3:
        st.info("Not enough history to forecast this series.")
        st.dataframe(monthly)
        return

    s = monthly.set_index("date")["quantity"].asfreq("MS").fillna(0.0)
    try:
        model = ExponentialSmoothing(
            s,
            seasonal_periods=12,
            trend="add",
            seasonal="add",
            initialization_method="estimated",
        )
        fit = model.fit()
        fc = fit.forecast(6)
        fc_df = pd.DataFrame({"month": fc.index, "forecast_quantity": fc.values})
    except Exception as e:
        st.warning(f"Forecast error: {e}")
        fc_df = pd.DataFrame()

    title = f"Monthly Sales (Units) - {sel}"
    fig = px.line(monthly, x="date", y="quantity", title=title)
    if not fc_df.empty:
        fig.add_scatter(
            x=fc_df["month"], y=fc_df["forecast_quantity"], name="Forecast", mode="lines"
        )
    st.plotly_chart(fig, use_container_width=True)


def page_campaign_roi(d: Dict[str, pd.DataFrame]):
    st.subheader("Campaign ROI")
    p = d["purchases"]
    t = d.get("traffic", pd.DataFrame())

    if p.empty:
        st.warning("No purchases data.")
        return

    rev = p.copy()
    rev["date"] = coerce_date(rev["date"])
    rev["channel"] = rev["platform"].fillna("Unknown")
    rev["date_only"] = rev["date"].dt.date
    by_rev = (
        rev.groupby(["date_only", "channel"], as_index=False).agg(
            revenue_attributed=("net_sales", "sum")
        )
    )
    by_rev["date"] = pd.to_datetime(by_rev["date_only"])
    by_rev = by_rev.drop(columns=["date_only"])

    if not t.empty:
        spend = t.copy()
        spend["date"] = coerce_date(
            spend.get("date", pd.Series(pd.NaT)), fallback=pd.Timestamp("2023-01-01")
        )
        if "channel" not in spend.columns:
            if "event_type" in spend.columns:
                spend["channel"] = spend["event_type"]
            elif "session_primary_channel_group" in spend.columns:
                spend["channel"] = spend["session_primary_channel_group"]
            else:
                spend["channel"] = "Unknown"
        if "spend" not in spend.columns:
            if "marketing_spend" in spend.columns:
                spend["spend"] = spend["marketing_spend"]
            else:
                spend["spend"] = 0.0
        spend["date_only"] = spend["date"].dt.date
        agg_spend = (
            spend.groupby(["date_only", "channel"], as_index=False).agg(
                spend=("spend", "sum")
            )
        )
        agg_spend["date"] = pd.to_datetime(agg_spend["date_only"])
        agg_spend = agg_spend.drop(columns=["date_only"])
    else:
        agg_spend = by_rev[["date", "channel"]].copy()
        agg_spend["spend"] = 0.0

    roi = agg_spend.merge(by_rev, on=["date", "channel"], how="left").fillna(
        {"revenue_attributed": 0.0}
    )
    if roi["spend"].sum() == 0:
        st.info("No marketing spend found; ROAS uses 0 spend fallback.")
    roi["ROAS"] = np.where(
        roi["spend"] > 0, roi["revenue_attributed"] / roi["spend"], np.nan
    )

    channels = sorted(roi["channel"].dropna().unique())
    sel = st.multiselect("Channel filter", channels, default=channels if channels else [])
    f = roi if not sel else roi[roi["channel"].isin(sel)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spend", f"${f['spend'].sum():,.0f}")
    c2.metric("Attributed Revenue", f"${f['revenue_attributed'].sum():,.0f}")
    c3.metric(
        "Avg ROAS",
        f"{(f['revenue_attributed'].sum() / f['spend'].sum()):.2f}"
        if f["spend"].sum()
        else "N/A",
    )

    if not f.empty:
        tline = f.groupby("date", as_index=False)[["spend", "revenue_attributed"]].sum()
        fig1 = px.line(
            tline,
            x="date",
            y=["spend", "revenue_attributed"],
            title="Spend vs Attributed Revenue (Daily)",
        )
        st.plotly_chart(fig1, use_container_width=True)

        tbar = f.groupby("channel", as_index=False)[["spend", "revenue_attributed"]].sum()
        fig2 = px.bar(
            tbar,
            x="channel",
            y=["spend", "revenue_attributed"],
            barmode="group",
            title="By Channel",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(f.sort_values(["date", "channel"]), use_container_width=True)


# ================== NLQ Page ==================
def page_nlq(d: Dict[str, pd.DataFrame]):
    st.subheader("NLQ Query")
    p = d["purchases"].copy()
    if p.empty:
        st.info("No purchases data to query.")
        return

    if "SKU_Description" not in p.columns:
        p["SKU_Description"] = p["sku"].astype(str)

    # Updated examples (Shopify instead of Shopee)
    st.caption(
        "Try: 'revenue shopify pistachios oct 2024', "
        "'revenue pistachios oct 2024', "
        "'orders lazada honey almonds jul 2024', "
        "'top five product sales'"
    )

    q = st.text_input("Query", "", key="nlq_query_input_v01f")

    def run_query(q: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        ql = q.lower().strip()
        res = df.copy()
        matched_any = False

        # -------- Platform / channel keywords --------
        channel_specified = False
        for ch in [
            "lazada","shopee","shopify","direct",
            "email","referral","organic search","paid search","social"
        ]:
            if ch in ql:
                matched_any = True
                channel_specified = True
                token = ch.split()[0]
                if ch in ["organic search","paid search","social","email","referral","direct"]:
                    res = res[res["platform"].str.lower().str.contains(token, na=False)]
                else:
                    res = res[res["platform"].str.lower() == token]

        if channel_specified and res.empty:
            st.warning("No matching data found. Please try another query.")
            return None

        # -------- Product keyword filter --------
        prod_keywords = [
            "honey almond","honey almonds","roasted cashew","roasted cashews",
            "natural cocktail mix","fancy mixed nuts","natural baked cashew","natural baked cashews",
            "natural baked almond","natural baked almonds","walnut butter","pistachio","pistachios",
            "cranberry","cranberries","dried mango","smoked almond","smoked almonds","satay cashew","satay cashews",
            "peanut","peanuts"
        ]
        matched = [k for k in prod_keywords if k in ql]
        if matched:
            matched_any = True
            pat = "|".join([re.escape(m) for m in matched])
            res = res[res["SKU_Description"].str.lower().str.contains(pat, na=False)]

        # -------- Date parsing --------
        months = {
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,
            "aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
            "january":1,"february":2,"march":3,"april":4,"june":6,
            "july":7,"august":8,"september":9,"october":10,
            "november":11,"december":12
        }
        m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
                      r"|january|february|march|april|june|july|august|"
                      r"september|october|november|december)\s+(\d{4})", ql)
        if m:
            matched_any = True
            mo = months[m.group(1)]
            yr = int(m.group(2))
            dcol = coerce_date(res["date"])
            res = res[(dcol.dt.month == mo) & (dcol.dt.year == yr)]
        else:
            y = re.search(r"\b(20\d{2})\b", ql)
            if y:
                matched_any = True
                yr = int(y.group(1))
                dcol = coerce_date(res["date"])
                res = res[dcol.dt.year == yr]

        if res.empty:
            st.warning("No matching data found. Please try another query.")
            return None

        # -------- Metrics tiles --------
        if "revenue" in ql or "sales" in ql:
            matched_any = True
            st.metric("Revenue", f"${res['net_sales'].sum():,.2f}")
        if "orders" in ql:
            matched_any = True
            ords = res["order_no"].nunique() if "order_no" in res.columns else len(res)
            st.metric("Orders", f"{ords:,}")
        if "qty" in ql or "units" in ql:
            matched_any = True
            st.metric("Units", f"{int(res['quantity'].sum()):,}")

        # -------- Top N queries --------
        topn_match = re.search(r"top\s+(\d+|five)\s+(?:product\s+)?sales", ql)
        if topn_match:
            matched_any = True
            n_raw = topn_match.group(1)
            n = int(n_raw) if n_raw.isdigit() else 5
            agg = (res.groupby("SKU_Description", dropna=False)["net_sales"].sum()
                     .reset_index().sort_values("net_sales", ascending=False))
            st.subheader(f"Top {n} SKUs by Sales")
            st.dataframe(agg.head(n).rename(columns={"net_sales":"sales"}), use_container_width=True)
            fig = px.bar(
                agg.head(n).rename(columns={"net_sales":"sales"}),
                x="SKU_Description", y="sales", title=f"Top {n} Sales"
            )
            st.plotly_chart(fig, use_container_width=True)
            return agg.head(n)

        if not matched_any:
            st.warning("We don't have enough data on that. Please check with the relevant department.")
            return None

        # -------- Output table --------
        cols_show = ["date","platform","SKU_Description","quantity","net_sales","campaign","customer_id"]
        cols_show = [c for c in cols_show if c in res.columns]
        out = res[cols_show].sort_values("date").rename(
            columns={"net_sales":"sales","customer_id":"Customer"}
        )
        return out

    if q:
        out = run_query(q, p)
        if out is not None and isinstance(out, pd.DataFrame) and not out.empty:
            st.dataframe(out, use_container_width=True)
    else:
        st.info("Enter a query and press Enter.")

# ================== Main ==================
def main():
    st.markdown(f"<h2 style='margin:0'>{APP_TITLE}</h2>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.sidebar.radio(
        "Go to",
        ["Home","Platform Performance","Product Insights","Forecasts","Campaign ROI","NLQ Query"]
    )
    d = load_all()

    if page == "Home":
        page_home(d)
    elif page == "Platform Performance":
        page_platform(d)
    elif page == "Product Insights":
        page_products(d)
    elif page == "Forecasts":
        page_forecasts(d)
    elif page == "Campaign ROI":
        page_campaign_roi(d)
    elif page == "NLQ Query":
        page_nlq(d)

if __name__ == "__main__":
    main()