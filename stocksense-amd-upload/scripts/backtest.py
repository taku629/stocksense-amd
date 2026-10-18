import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time; t=time.time()
from stocksense.data import generate
from stocksense.forecast import backtest
df=generate(); print(df.shape, df.units.mean())
r=backtest(df); print(r["table"]); print("cov",r["coverage_p10_p90"],"impr",r["improvement_vs_best_naive"], r["cutoff"], time.time()-t)
r["table"].to_csv(os.path.join(os.path.dirname(__file__), "..", "backtest_results.csv"),index=False)
