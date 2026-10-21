"""LightGBM quantile forecasting (P10/P50/P90) + backtest vs naive baselines."""
import numpy as np
import pandas as pd
import lightgbm as lgb

H = 28  # forecast horizon (days). Only lags >= H are used, so one model can forecast 28 days ahead.
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
FEATS = ["store_c", "sku_c", "cat_c", "dow", "month", "doy", "holiday", "promo", "price", "price_ratio",
         "lag_28", "lag_35", "lag_42", "lag_364", "rmean_28_28", "rmean_7_28", "rstd_28_28"]


def add_features(df):
    """Calendar, price/promo and lag features. Lags start at 28 days to avoid leakage within the horizon."""
    df = df.sort_values(["store", "sku", "date"]).copy()
    for c in ["store", "sku", "category"]:
        df[c[:3] + "_c" if c != "category" else "cat_c"] = df[c].astype("category").cat.codes
    df = df.rename(columns={"sto_c": "store_c"})
    df["dow"] = df.date.dt.dayofweek
    df["month"] = df.date.dt.month
    df["doy"] = df.date.dt.dayofyear
    df["price_ratio"] = df.price / df.base_price
    g = df.groupby(["store", "sku"]).units
    for l in [28, 35, 42, 364]:
        df[f"lag_{l}"] = g.shift(l)
    df["rmean_28_28"] = g.transform(lambda s: s.shift(28).rolling(28).mean())
    df["rmean_7_28"] = g.transform(lambda s: s.shift(28).rolling(7).mean())
    df["rstd_28_28"] = g.transform(lambda s: s.shift(28).rolling(28).std())
    return df


def train(df_feat, cutoff):
    """Train one LightGBM model per quantile on rows before cutoff."""
    tr = df_feat[(df_feat.date < cutoff) & df_feat.rmean_28_28.notna()]
    models = {}
    for name, a in QUANTILES.items():
        m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=400, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, verbose=-1)
        m.fit(tr[FEATS], tr.units, categorical_feature=["store_c", "sku_c", "cat_c"])
        models[name] = m
    return models


def predict(models, X):
    out = {k: np.clip(m.predict(X[FEATS]), 0, None) for k, m in models.items()}
    # keep quantiles ordered
    out["p10"] = np.minimum(out["p10"], out["p50"]); out["p90"] = np.maximum(out["p90"], out["p50"])
    return out


def wape(y, yhat):
    return float(np.abs(y - yhat).sum() / max(np.abs(y).sum(), 1e-9) * 100)


def backtest(df):
    """Hold out the last H days. Compare LightGBM P50 vs naive baselines with WAPE (%)."""
    f = add_features(df)
    cutoff = df.date.max() - pd.Timedelta(days=H - 1)
    models = train(f, cutoff)
    te = f[f.date >= cutoff].copy()
    p = predict(models, te)
    te["lgbm_p50"] = p["p50"]; te["p10"] = p["p10"]; te["p90"] = p["p90"]
    hist = f[f.date < cutoff]
    ma = hist[hist.date >= cutoff - pd.Timedelta(days=28)].groupby(["store", "sku"]).units.mean().rename("ma28")
    te = te.join(ma, on=["store", "sku"])
    last_wk = hist[hist.date >= cutoff - pd.Timedelta(days=7)][["store", "sku", "date", "units"]].copy()
    last_wk["dow"] = last_wk.date.dt.dayofweek
    te = te.merge(last_wk.rename(columns={"units": "last_week"})[["store", "sku", "dow", "last_week"]],
                  on=["store", "sku", "dow"], how="left")
    te["last_year"] = te["lag_364"]  # same weekday, 52 weeks earlier
    rows = []
    for name, col in [("LightGBM P50 (StockSense)", "lgbm_p50"), ("28-day moving average", "ma28"),
                      ("Same weekday last week", "last_week"), ("Same weekday last year", "last_year")]:
        rows.append({"model": name, "WAPE %": round(wape(te.units, te[col].fillna(0)), 1)})
    tbl = pd.DataFrame(rows)
    cover = float(((te.units >= te.p10) & (te.units <= te.p90)).mean() * 100)
    best_naive = tbl.iloc[1:]["WAPE %"].min()
    improvement = round((best_naive - tbl.iloc[0]["WAPE %"]) / best_naive * 100, 1)
    return {"table": tbl, "coverage_p10_p90": round(cover, 1), "improvement_vs_best_naive": improvement,
            "cutoff": cutoff, "test": te, "n_series": te.groupby(["store", "sku"]).ngroups}


def forecast_future(df, models=None, promo_days=None, discount=0.2, horizon=H):
    """Forecast the next `horizon` days after the data ends.
    promo_days: optional list of day offsets (0-based) to simulate a promotion (What-if)."""
    last = df.date.max()
    fut_dates = pd.date_range(last + pd.Timedelta(days=1), periods=horizon)
    from .data import jp_holidays
    hol = jp_holidays(fut_dates[0], fut_dates[-1])
    base = df[df.date == last][["store", "sku", "category", "base_price"]]
    fut = base.merge(pd.DataFrame({"date": fut_dates}), how="cross")
    fut["promo"] = 0
    fut["price"] = fut.base_price
    if promo_days:
        sel = fut.date.isin([fut_dates[i] for i in promo_days])
        fut.loc[sel, "promo"] = 1
        fut.loc[sel, "price"] = (fut.loc[sel, "base_price"] * (1 - discount)).round()
    fut["holiday"] = fut.date.isin(hol).astype(int)
    fut["units"] = np.nan
    allf = add_features(pd.concat([df, fut], ignore_index=True))
    if models is None:
        models = train(allf, last + pd.Timedelta(days=1))
    X = allf[allf.date > last].copy()
    p = predict(models, X)
    for k in p:
        X[k] = p[k]
    return X[["date", "store", "sku", "category", "price", "promo", "holiday", "p10", "p50", "p90"]], models
