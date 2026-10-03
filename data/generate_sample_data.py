"""Generate a messy retail dataset (missing values, outliers, duplicates) for demos."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 2000
regions = ["North", "South", "East", "West"]
cats = ["Electronics", "Clothing", "Home", "Beauty", "Sports"]

df = pd.DataFrame({
    "order_id": np.arange(1, n + 1),
    "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 540, n), unit="D"),
    "region": rng.choice(regions, n, p=[.3, .25, .25, .2]),
    "category": rng.choice(cats, n),
    "unit_price": rng.gamma(4, 25, n).round(2),
    "quantity": rng.integers(1, 8, n),
    "discount_pct": rng.choice([0, 5, 10, 15, 20], n, p=[.5, .2, .15, .1, .05]),
    "customer_age": rng.normal(38, 11, n).round(0),
})
df["revenue"] = (df.unit_price * df.quantity * (1 - df.discount_pct / 100)).round(2)
df["customer_rating"] = np.clip((3.2 + 0.015 * (df.customer_age - 38) - 0.02 * df.discount_pct
                                 + rng.normal(0, .8, n)), 1, 5).round(1)

# inject realistic problems
df.loc[rng.choice(n, 160, replace=False), "customer_age"] = np.nan
df.loc[rng.choice(n, 90, replace=False), "customer_rating"] = np.nan
df.loc[rng.choice(n, 40, replace=False), "region"] = np.nan
df.loc[rng.choice(n, 12, replace=False), "revenue"] *= 12            # outliers
df.loc[rng.choice(n, 6, replace=False), "customer_age"] = 140        # impossible ages
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)  # duplicates

df.to_csv("data/sample_retail_orders.csv", index=False)
print("Saved", df.shape)
