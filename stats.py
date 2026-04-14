import numpy as np
import pandas as pd


def calculate_stats(df):
    numeric_df = df.drop(columns=["id", "label"], errors="ignore")
    numeric_df = numeric_df.replace(-1, np.nan).astype(float)

    stats = []
    for col in numeric_df.columns:
        s = numeric_df[col].dropna()
        total = len(numeric_df[col])
        detected = s.count()

        stats.append({
            "landmark": col,
            "count": detected,
            "missing": total - detected,
            "missing_%": (total - detected) / total * 100,
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "iqr": s.quantile(0.75) - s.quantile(0.25),
            "skewness": s.skew(),
            "kurtosis": s.kurt(),
        })

    return pd.DataFrame(stats).set_index("landmark")


def calculate_stats_by_id(df, output_path="stats.csv"):
    rows = []

    for vid_id, group in df.groupby("id"):
        numeric_df = group.drop(columns=["id", "label"], errors="ignore")
        numeric_df = numeric_df.replace(-1, np.nan).astype(float)

        row = {"id": vid_id}
        for col in numeric_df.columns:
            row[f"{col}_min"] = numeric_df[col].min()

        rows.append(row)

    result = pd.DataFrame(rows).set_index("id")
    result.to_csv(output_path)
    return result

calculate_stats_by_id(pd.read_csv('vectors\\data.csv'))