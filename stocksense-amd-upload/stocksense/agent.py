"""Replenishment agent: an LLM on AMD Token Factory (OpenAI-compatible API) that calls Python tools.
All quantities are computed by tools; the LLM only orchestrates and explains.
A number guard checks every number in the final answer against tool outputs."""
import json, math, os, re, time
import pandas as pd


def amd_config(secrets=None):
    """Read AMD Token Factory settings from env vars or Streamlit secrets (never hard-coded)."""
    get = lambda k: os.environ.get(k) or ((secrets or {}).get(k) if secrets else None) or ""
    return {"base_url": get("AMD_TF_BASE_URL"), "api_key": get("AMD_TF_API_KEY"), "model": get("AMD_TF_MODEL")}


def pick_model(client, preferred=""):
    """Use AMD_TF_MODEL if given; else GET /models and prefer a Gemma model, else the first one."""
    if preferred:
        return preferred
    ids = [m.id for m in client.models.list().data]
    gem = [i for i in ids if "gemma" in i.lower()]
    return (gem or ids)[0]


class Tools:
    """Deterministic tools the LLM can call. `fc` = future forecast, `inv` = inventory snapshot."""
    def __init__(self, fc: pd.DataFrame, inv: pd.DataFrame, review_days: int = 7):
        self.fc, self.inv, self.review_days = fc, inv, review_days

    def get_forecast(self, store, sku, days=14):
        d = self.fc[(self.fc.store == store) & (self.fc.sku == sku)].sort_values("date").head(int(days))
        if d.empty:
            return {"error": f"unknown store/sku {store}/{sku}"}
        return {"store": store, "sku": sku, "days": int(days), "p10_total": round(float(d.p10.sum()), 1),
                "p50_total": round(float(d.p50.sum()), 1), "p90_total": round(float(d.p90.sum()), 1),
                "promo_days": int(d.promo.sum())}

    def get_inventory(self, store, sku):
        r = self.inv[(self.inv.store == store) & (self.inv.sku == sku)]
        if r.empty:
            return {"error": f"unknown store/sku {store}/{sku}"}
        r = r.iloc[0]
        return {k: (int(r[k]) if k not in ("store", "sku") else r[k]) for k in
                ["store", "sku", "on_hand", "on_order", "lead_time_days", "case_pack"]}

    def calc_order_qty(self, store, sku, safety_stock=0):
        """Order-up-to policy: target = P90 demand over (lead time + review period) + safety stock.
        order = target - on_hand - on_order, rounded UP to the case pack, never negative."""
        inv = self.get_inventory(store, sku)
        if "error" in inv:
            return inv
        cover = inv["lead_time_days"] + self.review_days
        f = self.get_forecast(store, sku, cover)
        target = math.ceil(f["p90_total"] + float(safety_stock))
        raw = max(0, target - inv["on_hand"] - inv["on_order"])
        qty = int(math.ceil(raw / inv["case_pack"]) * inv["case_pack"]) if raw > 0 else 0
        return {"store": store, "sku": sku, "cover_days": cover, "p50_demand": f["p50_total"],
                "p90_demand": f["p90_total"], "safety_stock": float(safety_stock), "order_up_to": target,
                "on_hand": inv["on_hand"], "on_order": inv["on_order"], "case_pack": inv["case_pack"],
                "order_qty": qty}

    def call(self, name, args):
        return getattr(self, name)(**args)

    def all_orders(self, store, safety_stock=0):
        """Deterministic order table for every SKU in a store (what the manager approves)."""
        return pd.DataFrame([self.calc_order_qty(store, s, safety_stock) for s in sorted(self.inv[self.inv.store == store].sku)])


TOOL_SPECS = [
    {"type": "function", "function": {"name": "get_forecast", "description": "Demand forecast totals (P10/P50/P90 units) for the next N days for one SKU at one store.",
     "parameters": {"type": "object", "properties": {"store": {"type": "string"}, "sku": {"type": "string"}, "days": {"type": "integer"}}, "required": ["store", "sku"]}}},
    {"type": "function", "function": {"name": "get_inventory", "description": "Current on-hand, on-order, supplier lead time and case pack for one SKU at one store.",
     "parameters": {"type": "object", "properties": {"store": {"type": "string"}, "sku": {"type": "string"}}, "required": ["store", "sku"]}}},
    {"type": "function", "function": {"name": "calc_order_qty", "description": "Compute the recommended order quantity (order-up-to policy using P90 demand over lead time + 7-day review period). Always use this for quantities.",
     "parameters": {"type": "object", "properties": {"store": {"type": "string"}, "sku": {"type": "string"}, "safety_stock": {"type": "number"}}, "required": ["store", "sku"]}}},
]

SYSTEM = ("You are StockSense, a replenishment assistant for a small retail store manager. "
          "Use the tools to get every number. Never compute or invent quantities yourself: the order quantity "
          "must be exactly calc_order_qty.order_qty. For each SKU you are asked about, call calc_order_qty "
          "(and get_forecast/get_inventory if useful), then write a short order draft: one line per SKU with "
          "the order quantity and a one-sentence reason citing the tool numbers (P90 demand, on-hand, lead time). "
          "Be concise. Final answer in English, plain text, no tables.")

NUM_RE = re.compile(r"(?<![A-Za-z])-?\d+(?:[.,]\d+)?")


def _nums(text):
    out = []
    for m in NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def number_guard(answer, tool_outputs, prompt):
    """Every number in the LLM answer must appear in a tool output or the user prompt (±0.6 rounding)."""
    allowed = set()
    for o in tool_outputs:
        allowed.update(_nums(json.dumps(o)))
    allowed.update(_nums(prompt))
    allowed.update(_nums(" ".join(o.get("sku", "") + " " + o.get("store", "") for o in tool_outputs if isinstance(o, dict))))
    bad = [n for n in _nums(answer) if not any(abs(n - a) <= 0.6 for a in allowed)]
    return {"passed": not bad, "checked": len(_nums(answer)), "unverified_numbers": bad}


def run_agent(cfg, tools: Tools, store, skus, max_turns=8):
    """Tool-calling loop against AMD Token Factory. Returns answer, tool log, model, latency, guard result."""
    from openai import OpenAI
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=90)
    model = pick_model(client, cfg.get("model"))
    prompt = f"Draft replenishment orders for store '{store}' for these SKUs: {', '.join(skus)}."
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    log, outputs, t0, llm_ms = [], [], time.time(), []
    answer = ""
    for _ in range(max_turns):
        t = time.time()
        r = client.chat.completions.create(model=model, messages=msgs, tools=TOOL_SPECS, tool_choice="auto", temperature=0)
        llm_ms.append(int((time.time() - t) * 1000))
        m = r.choices[0].message
        if not m.tool_calls:
            answer = m.content or ""
            break
        msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in m.tool_calls]})
        for c in m.tool_calls:
            try:
                args = json.loads(c.function.arguments or "{}")
                out = tools.call(c.function.name, args)
            except Exception as e:  # report tool errors back to the model
                args, out = c.function.arguments, {"error": str(e)}
            log.append({"tool": c.function.name, "args": args, "result": out})
            outputs.append(out)
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(out)})
    return {"answer": answer, "log": log, "model": model, "latency_s": round(time.time() - t0, 2),
            "llm_calls_ms": llm_ms, "guard": number_guard(answer, outputs, prompt)}
