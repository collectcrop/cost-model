import pandas as pd
import numpy as np

raw_text = """Index time:1170
Cache time:911073
Height: 4
IO time: 900065

Index time:1014
Cache time:1047617
Height: 4
IO time: 1036484

Index time:1435
Cache time:766928
Height: 4
IO time: 752020

Index time:1761
Cache time:23605758
Height: 4
IO time: 23569681

Index time:1790
Cache time:22962082
Height: 4
IO time: 22925381

Index time:1583
Cache time:923632
Height: 4
IO time: 905784

Index time:1157
Cache time:22875980
Height: 4
IO time: 22853506

Index time:1340
Cache time:1097144
Height: 4
IO time: 1085215

Index time:1386
Cache time:967993
Height: 4
IO time: 953888

Index time:1386
Cache time:967993
Height: 4
IO time: 953888
"""

# Parse the blocks
blocks = [b.strip() for b in raw_text.strip().split("\n\n") if b.strip()]
rows = []
for b in blocks:
    lines = b.splitlines()
    idx = int(lines[0].split(":")[1])
    cache = int(lines[1].split(":")[1])
    height = int(lines[2].split(":")[1])
    io = int(lines[3].split(":")[1])
    rows.append({"index_time": idx, "cache_time": cache, "height": height, "io_time": io})

df = pd.DataFrame(rows)
df.index = range(1, len(df)+1)

# Compute delta per formula: δ=(cacheTime-IOTime)*height/indexTime
df["delta"] = (df["cache_time"] - df["io_time"]) * df["height"] / df["index_time"]

# Robust outlier detection using Median Absolute Deviation (MAD)
median = np.median(df["delta"])
mad = np.median(np.abs(df["delta"] - median))

# Scale factor for normal consistency; avoid divide-by-zero
scale = 1.4826 if mad != 0 else 1.0
robust_z = (df["delta"] - median) / (scale * mad if mad != 0 else 1.0)
df["robust_z"] = robust_z

# Mark outliers with |robust_z| > 3
df["is_outlier"] = np.abs(df["robust_z"]) > 3

# Compute average of non-outliers
filtered = df[~df["is_outlier"]]
avg_delta = filtered["delta"].mean()

avg_delta, median, mad, filtered.shape[0], df.shape[0], df[df["is_outlier"]].index.tolist()

print(avg_delta,median)