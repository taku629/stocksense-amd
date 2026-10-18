import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, pandas as pd
from stocksense.data import generate; from stocksense.forecast import forecast_future
df = generate(); fc, _ = forecast_future(df)
s, k = "Tokyo-Shinjuku", "Green Tea 2L"
h = df[(df.store == s) & (df.sku == k)].tail(70); f = fc[(fc.store == s) & (fc.sku == k)]
hh = h.set_index("date").units.rolling(3, min_periods=1).mean()
fig = plt.figure(figsize=(19.2, 10.8), dpi=100); fig.patch.set_facecolor("#0b0d12")
ax = fig.add_axes([0.06, 0.08, 0.88, 0.5]); ax.set_facecolor("#0b0d12")
ax.plot(hh.index, hh.values, color="#9aa3b2", lw=3)
ax.fill_between(f.date, f.p10, f.p90, color="#ED1C24", alpha=0.25, lw=0)
ax.plot(f.date, f.p50, color="#ff3b42", lw=5)
ax.axvline(f.date.min(), color="#444a57", ls="--", lw=2)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
fig.text(0.06, 0.82, "StockSense — on AMD", color="white", fontsize=96, weight="bold")
fig.text(0.06, 0.73, "Demand forecasting (P10–P90) + replenishment agent on AMD Token Factory", color="#c9ced8", fontsize=31)
fig.text(0.06, 0.66, "LightGBM quantiles  ·  Gemma tool-calling agent  ·  manager approval", color="#ff6b70", fontsize=28)
fig.text(0.78, 0.6, "P10–P90 forecast", color="#ff6b70", fontsize=24)
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "submission/cover.png", facecolor=fig.get_facecolor())
