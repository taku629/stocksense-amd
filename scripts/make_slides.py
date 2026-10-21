"""Builds 6 slides as HTML, renders PNGs + a 16:9 PDF with headless Chromium."""
import os, base64
from playwright.sync_api import sync_playwright
S = os.path.join(os.path.dirname(__file__), "..", "submission"); S = os.path.abspath(S)
img = lambda p: "data:image/png;base64," + base64.b64encode(open(os.path.join(S, p), "rb").read()).decode()
CSS = """*{box-sizing:border-box;margin:0}body{margin:0;font-family:'DejaVu Sans',sans-serif}
.s{width:1920px;height:1080px;background:#0b0d12;color:#e8ebf0;padding:80px 110px;position:relative;overflow:hidden;page-break-after:always}
h1{font-size:72px;color:#fff;margin-bottom:12px}.k{color:#ff4d55;font-size:30px;letter-spacing:3px;text-transform:uppercase;margin-bottom:18px}
ul{font-size:40px;line-height:1.55;margin-top:30px;padding-left:44px}li{margin-bottom:10px}b{color:#ff6b70}
.f{position:absolute;bottom:40px;left:110px;right:110px;font-size:24px;color:#7d8594;display:flex;justify-content:space-between}
table{border-collapse:collapse;font-size:38px;margin-top:30px}td,th{padding:16px 34px;border-bottom:2px solid #2a2f3a;text-align:left}th{color:#9aa3b2}
.hl td{color:#fff;font-weight:bold;background:#3a1216}.box{border:3px solid #ff4d55;border-radius:18px;padding:22px 28px;font-size:30px;background:#151922}
.row{display:flex;gap:28px;align-items:center}.ar{font-size:50px;color:#ff4d55}"""
foot = lambda n: f'<div class="f"><span>StockSense — Takumu Hata · AMD Developer Hackathon ACT III · Reinvent Commerce</span><span>{n}/6</span></div>'
slides = [
f"""<div class=k>Problem</div><h1>Small retailers order by gut feeling</h1><ul>
<li>They lose margin <b>twice</b>: stockouts (lost sales) and overstock (waste, tied-up cash)</li>
<li>Fresh goods, promotions and holidays make demand swing day to day</li>
<li>Enterprise forecasting suites are too expensive and too opaque for a 3-store chain</li>
<li>Pure-LLM answers are not acceptable: quantities must be <b>computed and verifiable</b></li></ul>{foot(1)}""",
f"""<div class=k>Solution</div><h1>StockSense: forecast → agent → approval</h1><ul>
<li><b>LightGBM quantile models</b> forecast daily demand per SKU × store with P10 / P50 / P90 (weekday, holiday, price, promo)</li>
<li><b>Ordering agent</b> on AMD Token Factory calls tools <code>get_forecast</code>, <code>get_inventory</code>, <code>calc_order_qty</code> and explains each quantity</li>
<li><b>Number guard</b> checks every number the LLM writes against tool outputs</li>
<li><b>Manager approval gate</b> + edit log · <b>What-if</b>: "promo next week?"</li></ul>{foot(2)}""",
f"""<div class=k>Demo</div><h1 style="font-size:60px">Real run: Gemma-4-31B-IT on AMD Token Factory</h1>
<div class=row style="margin-top:26px;align-items:flex-start"><img src="{img('shots/1_forecast.png')}" style="width:830px;border-radius:12px">
<img src="{img('shots/3_agent.png')}" style="width:830px;border-radius:12px"></div>
<div style="font-size:30px;margin-top:22px;color:#c9ced8">Left: 28-day forecast with P10–P90 band. Right: agent order draft — 12 tool calls, number guard PASSED (21 numbers), model name + latency shown.</div>{foot(3)}""",
f"""<div class=k>Built on AMD</div><h1>All LLM inference on AMD GPUs</h1>
<div class=row style="margin-top:50px"><div class=box>Streamlit UI<br>(manager)</div><div class=ar>⇄</div>
<div class=box style="border-color:#fff">AMD Token Factory<br><b>Gemma-4-31B-IT</b><br>vLLM on AMD GPUs<br>OpenAI-compatible API</div><div class=ar>⇄</div>
<div class=box>Python tools<br>get_forecast · get_inventory<br>calc_order_qty</div><div class=ar>←</div><div class=box>LightGBM<br>P10/P50/P90</div></div>
<ul style="font-size:34px"><li>openai SDK with <code>base_url</code> = AMD Token Factory; model chosen via env or <code>GET /models</code> (prefers Gemma)</li>
<li>Measured: 2 LLM calls, 12 tool calls, 34 s end-to-end in the CLI run; latency + model shown in the UI</li>
<li>Keys only from env / Streamlit secrets — never in the repo</li></ul>{foot(4)}""",
f"""<div class=k>Results &amp; value</div><h1>14.2% lower WAPE than best naive</h1>
<div class=row style="align-items:flex-start;gap:70px"><table><tr><th>Model · 28-day holdout · 60 series</th><th>WAPE %</th></tr>
<tr class=hl><td>LightGBM P50 (StockSense)</td><td>35.0</td></tr><tr><td>28-day moving average</td><td>40.8</td></tr>
<tr><td>Same weekday last week</td><td>51.1</td></tr><tr><td>Same weekday last year</td><td>52.0</td></tr></table>
<ul style="font-size:32px;margin-top:40px"><li>75.4% of actuals inside P10–P90 band</li><li>Orders sized on P90 → fewer stockouts</li><li>Explained, approved orders → trust</li></ul></div>
<div style="font-size:26px;color:#9aa3b2;margin-top:34px">Synthetic data (20 SKUs × 3 stores × 2 years, weekday/holiday/price/promo effects). Measured on synthetic data, not a real-retailer claim.</div>{foot(5)}""",
f"""<div class=k>Next</div><h1>From demo to the store floor</h1><ul>
<li>Connect real POS feeds (Square, Shopify, Japanese POS) and validate on real sales</li>
<li>Multi-store transfers before placing new orders</li>
<li>Learn supplier lead-time distributions; spoilage-aware ordering for fresh goods</li>
<li>Send approved orders to suppliers (EDI / email) — still behind the approval gate</li></ul>
<div style="font-size:34px;margin-top:40px;color:#fff">github.com/taku629/stocksense-amd · MIT</div>{foot(6)}""",
]
html = f"<!doctype html><html><head><meta charset=utf-8><style>{CSS}</style></head><body>" + "".join(f'<div class=s>{s}</div>' for s in slides) + "</body></html>"
open(os.path.join(S, "slides.html"), "w").write(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox"])
    pg = b.new_page(viewport={"width": 1920, "height": 1080}); pg.set_content(html); pg.wait_for_timeout(800)
    for i, el in enumerate(pg.query_selector_all(".s")):
        el.screenshot(path=os.path.join(S, f"slide{i+1}.png"))
    pg.pdf(path=os.path.join(S, "stocksense-slides.pdf"), width="1920px", height="1080px", print_background=True,
           margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
    b.close()
