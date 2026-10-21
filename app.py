"""StockSense Streamlit app: forecast chart, backtest, AMD agent order drafts, approval gate, what-if."""
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stocksense.data import generate, inventory_snapshot, STORES
from stocksense.forecast import backtest, forecast_future
from stocksense.agent import Tools, amd_config, run_agent

st.set_page_config(page_title="StockSense on AMD", page_icon="📦", layout="wide")


@st.cache_resource(show_spinner="Generating synthetic sales data and training LightGBM quantile models…")
def load():
    df = generate()
    bt = backtest(df)
    fc, models = forecast_future(df)
    return df, bt, fc, models, inventory_snapshot(df)


df, bt, fc, models, inv = load()
try:
    secrets = dict(st.secrets)
except Exception:
    secrets = {}
cfg = amd_config(secrets)
agent_ready = bool(cfg["base_url"] and cfg["api_key"])

st.title("📦 StockSense — Demand Forecasting & Replenishment Agent on AMD")
st.caption("LightGBM quantile forecasts (P10/P50/P90) + an LLM ordering agent. "
           "**LLM inference runs on AMD GPUs via AMD Token Factory.** Data shown is synthetic.")

with st.sidebar:
    store = st.selectbox("Store", STORES)
    sku = st.selectbox("SKU", sorted(df.sku.unique()))
    st.markdown("---")
    st.markdown("**AMD Token Factory**")
    if agent_ready:
        st.success(f"Configured · model: `{cfg['model'] or 'auto (GET /models, prefer Gemma)'}`")
    else:
        st.warning("Agent needs the AMD key: set AMD_TF_BASE_URL, AMD_TF_API_KEY (and optionally AMD_TF_MODEL) "
                   "as env vars or Streamlit secrets. Forecasts still work without it.")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "🧪 Backtest", "🤖 Draft orders (AMD agent)", "🔮 What-if"])

# ---------- Forecast ----------
with tab1:
    hist = df[(df.store == store) & (df.sku == sku) & (df.date > df.date.max() - pd.Timedelta(days=90))]
    f = fc[(fc.store == store) & (fc.sku == sku)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.date, y=hist.units, name="Actual sales", line=dict(color="#888")))
    fig.add_trace(go.Scatter(x=list(f.date) + list(f.date[::-1]), y=list(f.p90) + list(f.p10[::-1]),
                             fill="toself", fillcolor="rgba(237,28,36,0.18)", line=dict(width=0), name="P10–P90 band"))
    fig.add_trace(go.Scatter(x=f.date, y=f.p50, name="Forecast P50", line=dict(color="#ED1C24", width=3)))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title=f"{sku} @ {store} — next 28 days",
                      yaxis_title="units / day")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Next 7 days P50", f"{f.head(7).p50.sum():.0f} units")
    c2.metric("Next 7 days P90", f"{f.head(7).p90.sum():.0f} units")
    r = inv[(inv.store == store) & (inv.sku == sku)].iloc[0]
    c3.metric("On hand / lead time", f"{r.on_hand} units / {r.lead_time_days} d")

# ---------- Backtest ----------
with tab2:
    st.subheader(f"Holdout: last 28 days (from {bt['cutoff']:%Y-%m-%d}), {bt['n_series']} SKU×store series")
    st.dataframe(bt["table"], hide_index=True, use_container_width=False)
    st.markdown(f"- WAPE reduction vs best naive baseline: **{bt['improvement_vs_best_naive']}%**\n"
                f"- Share of actuals inside the P10–P90 band: **{bt['coverage_p10_p90']}%** (nominal 80%)\n"
                "- WAPE = Σ|actual − forecast| / Σ actual. Lower is better. Synthetic data.")

# ---------- Agent ----------
with tab3:
    tools = Tools(fc, inv)
    st.markdown("Quantities are computed by **tools** (order-up-to with P90 demand over lead time + 7-day review). "
                "The LLM orchestrates tool calls and explains; a **number guard** checks every number it writes.")
    sel = st.multiselect("SKUs to draft", sorted(df.sku.unique()), default=sorted(df.sku.unique())[:4])
    if st.button("🤖 Draft orders with AMD agent", type="primary"):
        if not agent_ready:
            st.error("The agent needs the AMD Token Factory key (AMD_TF_BASE_URL / AMD_TF_API_KEY). "
                     "No LLM output is shown without it — forecasts and the tool-computed table below still work.")
        else:
            with st.spinner("Calling AMD Token Factory…"):
                try:
                    st.session_state.agent = run_agent(cfg, tools, store, sel)
                except Exception as e:
                    st.error(f"AMD Token Factory call failed: {e}")
    res = st.session_state.get("agent")
    if res:
        st.info(f"LLM inference runs on AMD GPUs via AMD Token Factory · model `{res['model']}` · "
                f"total latency {res['latency_s']} s · per-call {res['llm_calls_ms']} ms")
        st.markdown(res["answer"].replace("\n", "  \n"))
        g = res["guard"]
        (st.success if g["passed"] else st.warning)(
            f"Number guard: {'PASSED' if g['passed'] else 'CHECK'} — {g['checked']} numbers checked"
            + ("" if g["passed"] else f"; unverified: {g['unverified_numbers']}"))
        with st.expander(f"Tool-call log ({len(res['log'])} calls)"):
            for c in res["log"]:
                st.code(f"{c['tool']}({c['args']}) -> {c['result']}")
    # Approval gate: nothing is ordered automatically
    st.subheader("Order draft — manager approval")
    orders = tools.all_orders(store) if not sel else pd.DataFrame([tools.calc_order_qty(store, s) for s in sel])
    if not orders.empty:
        orders = orders[["sku", "on_hand", "on_order", "cover_days", "p90_demand", "case_pack", "order_qty"]]
        orders.insert(len(orders.columns), "approved_qty", orders.order_qty)
        edited = st.data_editor(orders, disabled=[c for c in orders.columns if c != "approved_qty"],
                                hide_index=True, key=f"ed_{store}")
        if st.button("✅ Approve orders"):
            hist_log = st.session_state.setdefault("edit_log", [])
            now = dt.datetime.now().strftime("%H:%M:%S")
            for _, row in edited.iterrows():
                if int(row.approved_qty) != int(row.order_qty):
                    hist_log.append({"time": now, "store": store, "sku": row.sku, "suggested": int(row.order_qty),
                                     "approved": int(row.approved_qty)})
            st.session_state.setdefault("approved", []).append({"time": now, "store": store,
                                                                "lines": int((edited.approved_qty > 0).sum()),
                                                                "units": int(edited.approved_qty.sum())})
            st.success(f"Approved {int((edited.approved_qty > 0).sum())} order lines "
                       f"({int(edited.approved_qty.sum())} units). Nothing is sent without this approval.")
        if st.session_state.get("edit_log"):
            st.caption("Manager edits (session history)")
            st.dataframe(pd.DataFrame(st.session_state.edit_log), hide_index=True)

# ---------- What-if ----------
with tab4:
    st.markdown(f"**What if we run a promotion next week?** Re-forecasts **{sku} @ {store}** with promo=1 and a discount on days 1–7.")
    disc = st.slider("Discount", 0.05, 0.40, 0.20, 0.05)
    if st.button("🔮 Re-forecast with promo next week"):
        wf, _ = forecast_future(df, models=models, promo_days=list(range(7)), discount=disc)
        a = fc[(fc.store == store) & (fc.sku == sku)].head(7)
        b = wf[(wf.store == store) & (wf.sku == sku)].head(7)
        comp = pd.DataFrame({"date": a.date.dt.strftime("%a %m-%d").values, "baseline P50": a.p50.round(1).values,
                             "promo P50": b.p50.round(1).values, "promo P90": b.p90.round(1).values})
        st.dataframe(comp, hide_index=True)
        t = Tools(wf, inv).calc_order_qty(store, sku)
        t0 = Tools(fc, inv).calc_order_qty(store, sku)
        st.metric("7-day P50 demand", f"{b.p50.sum():.0f} units", f"{b.p50.sum() - a.p50.sum():+.0f} vs baseline")
        st.metric("Recommended order", f"{t['order_qty']} units", f"{t['order_qty'] - t0['order_qty']:+d} vs baseline")
