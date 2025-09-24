import pandas as pd
import numpy as np

raw_text = """
Index time:1196
Cache time:2012700
Height: 4
IO time: 1886898

Index time:1124
Cache time:1171354
Height: 4
IO time: 1087567

Index time:1285
Cache time:1103254
Height: 4
IO time: 976470

Index time:1022
Cache time:23111522
Height: 4
IO time: 22987615

Index time:2349
Cache time:23950938
Height: 4
IO time: 23704946

Index time:1143
Cache time:1569841
Height: 4
IO time: 1483962

Index time:1140
Cache time:23673005
Height: 4
IO time: 23548751

Index time:1172
Cache time:23388117
Height: 4
IO time: 23209511

Index time:1220
Cache time:1604480
Height: 4
IO time: 1478618

Index time:1406
Cache time:839332
Height: 4
IO time: 640633

Index time:1444
Cache time:1031602
Height: 4
IO time: 918868
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
print(rows)
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