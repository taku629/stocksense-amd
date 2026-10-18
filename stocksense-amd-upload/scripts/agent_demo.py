import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import json
from stocksense.data import generate, inventory_snapshot
from stocksense.forecast import forecast_future
from stocksense.agent import Tools, run_agent
os.environ.setdefault("AMD_TF_BASE_URL","https://developer.amd.com.cn/radeon/api/v1")
cfg={"base_url":os.environ["AMD_TF_BASE_URL"],"api_key":os.environ["AMD_TF_API_KEY"],"model":os.environ.get("AMD_TF_MODEL") or "Gemma-4-31B-IT"}
df=generate(); fc,_=forecast_future(df); inv=inventory_snapshot(df)
r=run_agent(cfg,Tools(fc,inv),"Tokyo-Shinjuku",["Apple","Banana 1 bunch","Bread Loaf","Cabbage"])
print(json.dumps({k:v for k,v in r.items() if k!="log"},indent=1,default=str)); print(len(r["log"]),"tool calls"); print(r["log"][:2])
json.dump(r,open(os.path.join(os.path.dirname(__file__), "..", "agent_run_sample.json"),"w"),indent=1,default=str)
