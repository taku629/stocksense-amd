"""Synthetic (but realistic) retail sales generator.

NOTE: this data is SYNTHETIC. It mimics typical small-retail patterns
(weekday seasonality, holidays, price elasticity, promotions, yearly season,
trend, Poisson/negative-binomial noise). No real store data is used.
"""
import numpy as np
import pandas as pd

STORES = ["Tokyo-Shinjuku", "Osaka-Umeda", "Fukuoka-Tenjin"]
SKUS = [  # (sku, category, base daily units, base price JPY)
    ("Rice 5kg", "Grocery", 6, 2480), ("Eggs 10pk", "Grocery", 22, 298),
    ("Milk 1L", "Dairy", 35, 238), ("Yogurt 400g", "Dairy", 14, 198),
    ("Bread Loaf", "Bakery", 18, 168), ("Croissant 4pk", "Bakery", 7, 358),
    ("Green Tea 2L", "Beverage", 20, 178), ("Cola 1.5L", "Beverage", 12, 198),
    ("Coffee Beans 200g", "Beverage", 4, 980), ("Instant Ramen 5pk", "Grocery", 10, 498),
    ("Natto 3pk", "Grocery", 16, 98), ("Tofu 300g", "Grocery", 15, 88),
    ("Banana 1 bunch", "Produce", 25, 198), ("Apple", "Produce", 9, 158),
    ("Cabbage", "Produce", 8, 248), ("Chicken Thigh 300g", "Meat", 11, 448),
    ("Salmon Fillet", "Seafood", 5, 598), ("Ice Cream Bar", "Frozen", 13, 128),
    ("Frozen Gyoza", "Frozen", 6, 348), ("Toilet Paper 12R", "Household", 4, 598),
]
WEEKDAY = {"Grocery": [1.0, .95, .95, 1.0, 1.1, 1.35, 1.25], "Dairy": [1.05, 1, 1, 1, 1.05, 1.2, 1.1],
           "Bakery": [1, .95, .95, 1, 1.1, 1.4, 1.3], "Beverage": [.95, .95, .95, 1, 1.1, 1.35, 1.3],
           "Produce": [1, .95, 1, 1, 1.05, 1.3, 1.2], "Meat": [.95, .9, .95, 1, 1.15, 1.4, 1.3],
           "Seafood": [.9, .9, .95, 1, 1.2, 1.45, 1.3], "Frozen": [1, 1, 1, 1, 1.05, 1.25, 1.2],
           "Household": [.9, .9, .9, .95, 1, 1.45, 1.4]}
SUMMER = {"Beverage": 0.35, "Frozen": 0.6, "Produce": 0.05}  # amplitude of yearly season


def jp_holidays(start, end):
    """Small fixed list of Japanese public holidays + year-end (approximate, good enough for synthetic data)."""
    md = ["01-01", "01-02", "01-03", "01-13", "02-11", "02-23", "03-20", "04-29", "05-03", "05-04", "05-05",
          "05-06", "07-21", "08-11", "08-13", "08-14", "08-15", "09-15", "09-23", "10-13", "11-03", "11-23",
          "12-29", "12-30", "12-31"]
    yrs = range(pd.Timestamp(start).year, pd.Timestamp(end).year + 2)
    return set(pd.Timestamp(f"{y}-{d}") for y in yrs for d in md)


def generate(start="2024-10-01", days=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    hol = jp_holidays(dates[0], dates[-1])
    is_hol = np.array([d in hol for d in dates])
    doy = dates.dayofyear.values
    rows = []
    for si, store in enumerate(STORES):
        store_mult = [1.3, 1.0, 0.75][si]
        for sku, cat, base, price in SKUS:
            b = base * store_mult * rng.uniform(0.85, 1.15)
            trend = 1 + rng.uniform(-0.08, 0.15) * np.arange(days) / 365
            season = 1 + SUMMER.get(cat, rng.uniform(0, .05)) * np.sin(2 * np.pi * (doy - 110) / 365)
            wd = np.array(WEEKDAY[cat])[dates.dayofweek.values]
            hol_eff = np.where(is_hol, rng.uniform(1.15, 1.5), 1.0)
            # promotions: ~week-long campaigns a few times per quarter, store-specific
            promo = np.zeros(days, dtype=int)
            for s in rng.choice(days - 7, size=int(days / 45), replace=False):
                promo[s:s + rng.integers(3, 8)] = 1
            disc = np.where(promo == 1, rng.choice([.1, .15, .2, .3]), 0.0)
            # occasional regular price changes (±5%)
            price_path = np.full(days, float(price))
            for c in rng.choice(days, size=2, replace=False):
                price_path[c:] *= rng.choice([0.95, 1.05])
            sell_price = np.round(price_path * (1 - disc))
            elasticity = rng.uniform(1.2, 2.5)
            price_eff = (sell_price / price) ** (-elasticity)
            promo_lift = np.where(promo == 1, rng.uniform(1.1, 1.3), 1.0)  # display/flyer effect
            lam = b * trend * season * wd * hol_eff * price_eff * promo_lift
            # negative-binomial noise (overdispersed like real retail)
            k = 8.0
            units = rng.poisson(rng.gamma(k, lam / k))
            rows.append(pd.DataFrame({"date": dates, "store": store, "sku": sku, "category": cat,
                                      "price": sell_price, "base_price": price_path, "promo": promo,
                                      "holiday": is_hol.astype(int), "units": units}))
    return pd.concat(rows, ignore_index=True)


def inventory_snapshot(df, seed=7):
    """Synthetic current inventory per SKU x store: on-hand, on-order, lead time, case pack."""
    rng = np.random.default_rng(seed)
    last = df[df.date > df.date.max() - pd.Timedelta(days=28)].groupby(["store", "sku"]).units.mean()
    inv = []
    for (store, sku), avg in last.items():
        inv.append({"store": store, "sku": sku,
                    "on_hand": int(avg * rng.uniform(1.0, 8.0)),
                    "on_order": int(rng.choice([0, 0, 0, avg * 3])),
                    "lead_time_days": int(rng.choice([1, 2, 3, 5, 7])),
                    "case_pack": int(rng.choice([1, 6, 12])) if avg > 5 else 1})
    return pd.DataFrame(inv)
