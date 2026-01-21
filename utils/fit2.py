import os
import numpy as np
import pandas as pd

from optimalEpsilon import cost_function, range_cost_function

# DATASETS_DIRECTORY = "/mnt/home/zwshi/Datasets/SOSD/"
DATASETS_DIRECTORY = "/mnt/backup_disk/backup_2025_full/zwshi/Datasets/SOSD/"
REAL_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/include/FALCON/results/log/"
LOG_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/visualize/data/log/"


# ============================================================
# 0) 统一口径：本脚本中所有 cost / avg_IOs 都表示
#    "average I/Os per query"（每 query 平均 I/O 次数）
# ============================================================

def mib_to_bytes(M_mib: float) -> float:
    return float(M_mib) * 1024.0 * 1024.0


# ============================================================
# 1) 读取真实测量（real）: epsilon -> avg_IOs (per query)
# ============================================================

def load_real_avg_ios_per_query(csv_path: str, max_eps: int = 64) -> pd.DataFrame:
    """
    从 real CSV 中读取每个 epsilon 的 avg_IOs 均值。
    返回 DataFrame: [epsilon, avg_IOs]，其中 avg_IOs 是 "per query average I/Os"
    """
    df = pd.read_csv(csv_path)
    if "epsilon" not in df.columns or "avg_IOs" not in df.columns:
        raise ValueError(f"{csv_path} must contain columns: epsilon, avg_IOs")

    grouped = df.groupby("epsilon", as_index=False)["avg_IOs"].mean()
    grouped = grouped[grouped["epsilon"] <= max_eps].copy()
    grouped["epsilon"] = grouped["epsilon"].astype(float)
    grouped["avg_IOs"] = grouped["avg_IOs"].astype(float)
    return grouped


def normalize_to_per_query(values: np.ndarray, *, is_total: bool, num_queries: int) -> np.ndarray:
    """
    将 values 统一成 per-query：
      - 若 is_total=True，则 values / num_queries
      - 否则不变
    """
    values = np.asarray(values, dtype=np.float64)
    if is_total:
        if num_queries <= 0:
            raise ValueError("num_queries must be >0 when is_total=True")
        return values / float(num_queries)
    return values


# ============================================================
# 2) 计算模型预测（estimated/model）: epsilon -> cost_hat (per query)
# ============================================================

def compute_model_costs_per_query(
    M_mib: float,
    eps_list: np.ndarray,
    *,
    n: int,
    seg_size: int,
    ipp: int,
    ps: int,
    type: str,
    data_file: str,
    query_file: str,
    fetch_strategy: str = "all_in_once",
    mode: str = "point",
) -> np.ndarray:
    """
    对固定 M（MiB）与一组 eps，调用 cost_function/range_cost_function，
    返回 cost_hat（per-query average I/Os）。
    """
    M_bytes = mib_to_bytes(M_mib)
    data_path = f"{DATASETS_DIRECTORY}{data_file}"
    query_path = f"{DATASETS_DIRECTORY}{query_file}"

    eps_arr = np.asarray(eps_list, dtype=np.float64)
    cost_hat = np.zeros_like(eps_arr, dtype=np.float64)

    for i, eps in enumerate(eps_arr):
        if mode == "point":
            c, _h = cost_function(
                eps,
                n,
                seg_size,
                M_bytes,
                ipp,
                ps,
                type=type,
                query_file=query_path,
                data_file=data_path,
                s=fetch_strategy,
            )
        elif mode == "range":
            c, _h = range_cost_function(
                eps,
                n,
                seg_size,
                M_bytes,
                ipp,
                ps,
                query_file=query_path,
                data_file=data_path,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        cost_hat[i] = float(c)

    return cost_hat


# ============================================================
# 3) 拟合加性残差模型： r(eps,M) ≈ y_real - y_hat
#    使用可解释基函数：1, M, eps, inv, M*eps, M*inv
# ============================================================

def build_additive_residual_features(eps: np.ndarray, M_mib: np.ndarray, eps0: float = 1.0) -> np.ndarray:
    """
    设计矩阵 X，使得：
      r = a0 + a1*M + a2*eps + a3*inv + a4*M*eps + a5*M*inv
    inv = 1/(eps + eps0)
    """
    eps = np.asarray(eps, dtype=np.float64)
    M_mib = np.asarray(M_mib, dtype=np.float64)
    inv = 1.0 / (eps + eps0)

    X = np.column_stack([
        np.ones_like(eps),
        M_mib,
        eps,
        inv,
        M_mib * eps,
        M_mib * inv,
    ])
    return X


    
def fit_ridge_with_standardization(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1e-2) -> np.ndarray:
    """
    对除截距列外的特征做 z-score 标准化，然后做岭回归，最后还原到原尺度。
    返回原尺度系数 coef，使得 y ≈ X @ coef
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    Xs = X.copy()
    mu = Xs[:, 1:].mean(axis=0)
    sigma = Xs[:, 1:].std(axis=0) + 1e-12
    Xs[:, 1:] = (Xs[:, 1:] - mu) / sigma

    XtX = Xs.T @ Xs
    I = np.eye(XtX.shape[0], dtype=np.float64)
    coef_std = np.linalg.solve(XtX + ridge_lambda * I, Xs.T @ y)

    # 还原到原尺度
    coef = np.zeros_like(coef_std)
    coef[1:] = coef_std[1:] / sigma
    coef[0] = coef_std[0] - np.sum(coef_std[1:] * mu / sigma)
    return coef


def fit_additive_residual_model_global(
    Ms_mib: list,
    real_csv_files: list,
    *,
    n: int,
    seg_size: int,
    ipp: int,
    ps: int,
    type: str,
    data_file: str,
    query_file: str,
    fetch_strategy: str = "all_in_once",
    mode: str = "point",
    max_eps: int = 64,
    eps0: float = 1.0,
    ridge_lambda: float = 1e-2,
    # 下面两个开关用于“口径对齐”
    real_is_total: bool = False,
    num_queries: int = 0,
) -> np.ndarray:
    """
    用多个内存预算的数据拼接起来拟合全局残差模型 r(eps,M)。

    注意：本函数严格在 "per-query avg I/Os" 口径上拟合。
    - 如果你的 real CSV 的 avg_IOs 实际是 total I/Os，设 real_is_total=True 并提供 num_queries
    - 否则保持默认 False
    """
    if len(Ms_mib) != len(real_csv_files):
        raise ValueError("Ms_mib and real_csv_files must have same length")

    all_X, all_r = [], []

    for M_mib, csv_path in zip(Ms_mib, real_csv_files):
        real_df = load_real_avg_ios_per_query(csv_path, max_eps=max_eps)
        eps = real_df["epsilon"].to_numpy(dtype=np.float64)
        y_real = real_df["avg_IOs"].to_numpy(dtype=np.float64)
        y_real = normalize_to_per_query(y_real, is_total=real_is_total, num_queries=num_queries)

        y_hat = compute_model_costs_per_query(
            M_mib, eps,
            n=n, seg_size=seg_size, ipp=ipp, ps=ps,
            type=type,
            data_file=data_file,
            query_file=query_file,
            fetch_strategy=fetch_strategy,
            mode=mode,
        )

        # 加性残差（per query）
        r = y_real - y_hat

        M_vec = np.full_like(eps, float(M_mib), dtype=np.float64)
        X = build_additive_residual_features(eps, M_vec, eps0=eps0)

        all_X.append(X)
        all_r.append(r)

    X_all = np.vstack(all_X)
    r_all = np.concatenate(all_r)

    coef = fit_ridge_with_standardization(X_all, r_all, ridge_lambda=ridge_lambda)
    return coef


def predict_additive_residual(eps: np.ndarray, M_mib: np.ndarray, coef: np.ndarray, eps0: float = 1.0) -> np.ndarray:
    eps = np.asarray(eps, dtype=np.float64)
    M_mib = np.asarray(M_mib, dtype=np.float64)
    X = build_additive_residual_features(eps, M_mib, eps0=eps0)
    return X @ np.asarray(coef, dtype=np.float64)


# ============================================================
# 4) 应用修正到 estimated log：输出 revision log
#    input/output 表头都为：M,epsilon,cost,ratio
# ============================================================

def apply_additive_correction_to_estimated_log(
    input_log_path: str,
    output_log_path: str,
    coef: np.ndarray,
    *,
    eps0: float = 1.0,
    residual_clip: tuple | None = None,
    # 口径对齐开关
    estimated_is_total: bool = False,
    num_queries: int = 0,
    assume_M_unit: str = "MiB",  # 一般你的 log 是 10/20/40/60 -> MiB
    ratio_field: str = "residual",  # "residual" 或 "keep_original"
):
    """
    从 estimated log 读取 (M,epsilon,cost,ratio)。
    默认认为 cost 是 per-query average I/Os；若 cost 是 total I/Os，请设 estimated_is_total=True 并给 num_queries。

    输出 revision log：
      cost := cost + r_pred
      ratio := r_pred（默认），或保留原 ratio（若 ratio_field="keep_original"）
    """
    df = pd.read_csv(input_log_path)
    required = {"M", "epsilon", "cost", "ratio"}
    if not required.issubset(df.columns):
        raise ValueError(f"{input_log_path} must have columns {required}")

    M_raw = df["M"].to_numpy(dtype=np.float64)
    eps = df["epsilon"].to_numpy(dtype=np.float64)
    y_hat = df["cost"].to_numpy(dtype=np.float64)

    # M 单位处理（通常就是 MiB）
    if assume_M_unit == "bytes":
        M_mib = M_raw / (1024.0 * 1024.0)
    else:
        M_mib = M_raw

    # cost 口径统一到 per-query
    y_hat = normalize_to_per_query(y_hat, is_total=estimated_is_total, num_queries=num_queries)

    # 预测残差并修正
    r_pred = predict_additive_residual(eps, M_mib, coef, eps0=eps0)
    if residual_clip is not None:
        r_pred = np.clip(r_pred, residual_clip[0], residual_clip[1])

    y_rev = y_hat + r_pred

    # 输出时保持和输入一致口径：如果输入是 total，则输出也写回 total
    if estimated_is_total:
        y_rev_out = y_rev * float(num_queries)
    else:
        y_rev_out = y_rev

    if ratio_field == "keep_original":
        ratio_out = df["ratio"]
    else:
        ratio_out = r_pred  # 推荐：ratio 记录加性补偿量

    out = pd.DataFrame({
        "M": df["M"],
        "epsilon": df["epsilon"],
        "cost": y_rev_out,
        "ratio": ratio_out,
    })

    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    out.to_csv(output_log_path, index=False, float_format="%.10g")
    print(f"[OK] wrote revised log: {output_log_path}")


def evaluate_holdout_memory_budget(
    M_holdout_mib: float,
    real_csv_path: str,
    coef: np.ndarray,
    *,
    n: int,
    seg_size: int,
    ipp: int,
    ps: int,
    type: str,
    data_file: str,
    query_file: str,
    fetch_strategy: str = "all_in_once",
    mode: str = "point",
    max_eps: int = 64,
    eps0: float = 1.0,
    # 口径对齐
    real_is_total: bool = False,
    num_queries: int = 0,
):
    """
    用未参与拟合的 hold-out (M=30MiB) 做验证：
      - y_real: per-query avg I/Os（必要时从 total 归一化）
      - y_hat:  模型预测（per-query）
      - y_corr: y_hat + r_pred

    输出多种误差指标，便于判断是否过拟合。
    """
    real_df = load_real_avg_ios_per_query(real_csv_path, max_eps=max_eps)
    eps = real_df["epsilon"].to_numpy(dtype=np.float64)
    y_real = real_df["avg_IOs"].to_numpy(dtype=np.float64)
    y_real = normalize_to_per_query(y_real, is_total=real_is_total, num_queries=num_queries)

    y_hat = compute_model_costs_per_query(
        M_holdout_mib, eps,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy=fetch_strategy,
        mode=mode,
    )

    # 校正：加性残差
    M_vec = np.full_like(eps, float(M_holdout_mib), dtype=np.float64)
    r_pred = predict_additive_residual(eps, M_vec, coef, eps0=eps0)
    y_corr = y_hat + r_pred

    # 误差指标
    err_before = y_real - y_hat
    err_after = y_real - y_corr

    mae_before = float(np.mean(np.abs(err_before)))
    mae_after  = float(np.mean(np.abs(err_after)))

    rmse_before = float(np.sqrt(np.mean(err_before**2)))
    rmse_after  = float(np.sqrt(np.mean(err_after**2)))

    # 相对误差（用 y_real 做分母更直观；也可用 y_hat）
    denom = np.maximum(y_real, 1e-12)
    rel_before = err_before / denom
    rel_after  = err_after / denom

    mean_abs_rel_before = float(np.mean(np.abs(rel_before)))
    mean_abs_rel_after  = float(np.mean(np.abs(rel_after)))

    max_abs_rel_before = float(np.max(np.abs(rel_before)))
    max_abs_rel_after  = float(np.max(np.abs(rel_after)))

    print("\n========== Hold-out Validation ==========")
    print(f"M_holdout = {M_holdout_mib} MiB, file = {real_csv_path}")
    print(f"MAE   : before={mae_before:.6g}, after={mae_after:.6g}")
    print(f"RMSE  : before={rmse_before:.6g}, after={rmse_after:.6g}")
    print(f"Mean|rel|: before={mean_abs_rel_before:.6g}, after={mean_abs_rel_after:.6g}")
    print(f"Max |rel|: before={max_abs_rel_before:.6g}, after={max_abs_rel_after:.6g}")

# ============================================================
# 5) Demo：books_10M point query，全局拟合 + 修正 log
# ============================================================

def run_books10m_additive_calibration_and_revision():
    # ---- 实验与模型参数 ----
    n = int(1e7)
    seg_size = 16
    ipp = 512
    ps = 4096
    type = "sample"

    data_file = "books_10M_uint64_unique"
    query_file = "books_10M_uint64_unique.query.bin"

    # ---- 用这些 M 的真实数据拟合 ----
    Ms = [10, 20, 40, 60]
    real_csv_files = [
        os.path.join(REAL_DIRECTORY, "books_10M_M10_falcon.csv"),
        os.path.join(REAL_DIRECTORY, "books_10M_M20_falcon.csv"),
        os.path.join(REAL_DIRECTORY, "books_10M_M40_falcon.csv"),
        os.path.join(REAL_DIRECTORY, "books_10M_M60_falcon.csv"),
    ]

    # 关键：如果你已经把 real/estimated 都修正成 per-query，请保持 False
    real_is_total = True
    num_queries = (int)(7e5)  # 仅当 real_is_total=True 才需要

    coef = fit_additive_residual_model_global(
        Ms_mib=Ms,
        real_csv_files=real_csv_files,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy="all_in_once",
        mode="point",
        max_eps=64,
        eps0=0.0,
        ridge_lambda=1e-2,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    print("Fitted additive residual coef [a0,a1,a2,a3,a4,a5]:")
    print(coef)

    holdout_M = 30
    holdout_csv = os.path.join(REAL_DIRECTORY, "books_10M_M30_falcon.csv")  # 按你真实文件名调整

    evaluate_holdout_memory_budget(
        M_holdout_mib=holdout_M,
        real_csv_path=holdout_csv,
        coef=coef,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy="all_in_once",
        mode="point",
        max_eps=64,
        eps0=0.0,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    # ---- 修正 estimated log ----
    in_log = os.path.join(LOG_DIRECTORY, "books_10M_uint64_unique.query.log")
    out_log = os.path.join(LOG_DIRECTORY, "books_10M_uint64_unique_revision.query.log")

    estimated_is_total = False
    num_queries_est = 0  # 仅当 estimated_is_total=True 才需要

    apply_additive_correction_to_estimated_log(
        input_log_path=in_log,
        output_log_path=out_log,
        coef=coef,
        eps0=0.0,
        residual_clip=None,          
        estimated_is_total=estimated_is_total,
        num_queries=num_queries_est,
        assume_M_unit="MiB",
        ratio_field="residual",      # ratio 写 r_pred；若你要保留原 ratio 改 "keep_original"
    )


if __name__ == "__main__":
    run_books10m_additive_calibration_and_revision()
