import os, re, glob
import numpy as np
import pandas as pd

REAL_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/include/FALCON/results/log/"

def extract_N(path: str) -> int | None:
    m = re.search(r'(\d+)Mquery', os.path.basename(path))
    return int(m.group(1)) if m else None

def extract_N2(path: str):
    m = re.search(r'join(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None

def load_point_runs(files, epsilon=16):
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d["epsilon"] == epsilon].copy()
        d["file"] = os.path.basename(f)
        d["N"] = extract_N(f)*(1e6)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_range_runs(files, epsilon=16):
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        d = d[d["epsilon"] == epsilon].copy()
        d["file"] = os.path.basename(f)
        d["N"] = extract_N2(f)*(1e6)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def fit_point():
    files = sorted(glob.glob(f"{REAL_DIRECTORY}books_200M_*Mquery_join.point.csv"))
    df = load_point_runs(files, epsilon=16)

    IO_COL = "IOs" if "IOs" in df.columns else "avg_IOs"
    df["lambda"] = df["IO_time_s"] / df[IO_COL]
    q1, q3 = df["lambda"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df2 = df[(df["lambda"] >= q1 - 1.5*iqr) & (df["lambda"] <= q3 + 1.5*iqr)]

    lambda_point = df2["lambda"].median()
    print("lambda_point(s/page)=", lambda_point, "=", lambda_point*1e6, "us/page")

    # df["alpha_sample"] = (df["total_wall_time_s"] - lambda_point * df[IO_COL]) / df["N"]
    df["T_cpu"] = (df["total_wall_time_s"] - lambda_point * df[IO_COL])
    alpha,delta = np.polyfit(df["N"].values, df["T_cpu"].values, 1)   
    print("alpha=", alpha)
    print("delta=", delta)
    
def fit_range():
    files = sorted(glob.glob(f"{REAL_DIRECTORY}books_200M_query_join*.range.csv"))
    df = load_range_runs(files, epsilon=16)

    IO_COL = "IOs" if "IOs" in df.columns else "avg_IOs"
    df["lambda"] = df["IO_time_s"] / df[IO_COL]
    df["K"] = df[IO_COL]
    q1, q3 = df["lambda"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df2 = df[(df["lambda"] >= q1 - 1.5*iqr) & (df["lambda"] <= q3 + 1.5*iqr)]

    lambda_range = df2["lambda"].median()
    print("lambda_range(s/page)=", lambda_range, "=", lambda_range*1e6, "us/page")

    df["T_cpu"] = (df["total_wall_time_s"] - lambda_range * df[IO_COL])
    beta,eta = np.polyfit(df["K"].values, df["T_cpu"].values, 1)   
    print("beta(s/page_scan)=", beta, "eta(s)=", eta) 


