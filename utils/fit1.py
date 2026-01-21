# residual_linear_fit.py
import os
import numpy as np
import pandas as pd

# ============================================================
# IMPORTANT: set this to your original module filename (without .py)
# Example: if your original file is named "cam_residual.py",
# set BASE_MODULE_NAME = "cam_residual"
# ============================================================
BASE_MODULE_NAME = "fit2"  

base = __import__(BASE_MODULE_NAME)


# ============================================================
# Linear residual model: r = B + k1*epsilon + k2*M
# ============================================================

def build_linear_residual_features(eps: np.ndarray, M_mib: np.ndarray) -> np.ndarray:
    """
    Design matrix for: r = B + k1*eps + k2*M
    X = [1, eps, M]
    """
    eps = np.asarray(eps, dtype=np.float64)
    M_mib = np.asarray(M_mib, dtype=np.float64)
    return np.column_stack([np.ones_like(eps), eps, M_mib])


def fit_linear_residual_model_global(
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
    ridge_lambda: float = 0.0,
    # align measurement scale
    real_is_total: bool = False,
    num_queries: int = 0,
) -> np.ndarray:
    """
    Fit residual r(eps,M) = y_real - y_hat with a simple linear model:
        r = B + k1*eps + k2*M

    Returns:
        coef = [B, k1, k2]
    """
    if len(Ms_mib) != len(real_csv_files):
        raise ValueError("Ms_mib and real_csv_files must have same length")

    all_X, all_r = [], []

    for M_mib, csv_path in zip(Ms_mib, real_csv_files):
        # real: avg I/Os (per query), unless real_is_total=True
        real_df = base.load_real_avg_ios_per_query(csv_path, max_eps=max_eps)
        eps = real_df["epsilon"].to_numpy(dtype=np.float64)
        y_real = real_df["avg_IOs"].to_numpy(dtype=np.float64)
        y_real = base.normalize_to_per_query(y_real, is_total=real_is_total, num_queries=num_queries)

        # model prediction (per query)
        y_hat = base.compute_model_costs_per_query(
            M_mib, eps,
            n=n, seg_size=seg_size, ipp=ipp, ps=ps,
            type=type,
            data_file=data_file,
            query_file=query_file,
            fetch_strategy=fetch_strategy,
            mode=mode,
        )

        r = y_real - y_hat
        M_vec = np.full_like(eps, float(M_mib), dtype=np.float64)

        X = build_linear_residual_features(eps, M_vec)
        all_X.append(X)
        all_r.append(r)

    X_all = np.vstack(all_X)
    r_all = np.concatenate(all_r)

    if ridge_lambda > 0.0:
        XtX = X_all.T @ X_all
        I = np.eye(XtX.shape[0], dtype=np.float64)
        coef = np.linalg.solve(XtX + ridge_lambda * I, X_all.T @ r_all)
    else:
        coef, *_ = np.linalg.lstsq(X_all, r_all, rcond=None)

    return coef  # [B, k1, k2]


def predict_linear_residual(eps: np.ndarray, M_mib: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """
    coef = [B, k1, k2]
    r = B + k1*eps + k2*M
    """
    eps = np.asarray(eps, dtype=np.float64)
    M_mib = np.asarray(M_mib, dtype=np.float64)
    X = build_linear_residual_features(eps, M_mib)
    return X @ np.asarray(coef, dtype=np.float64)


def apply_linear_residual_to_estimated_log(
    input_log_path: str,
    output_log_path: str,
    coef: np.ndarray,
    *,
    residual_clip: tuple | None = None,
    estimated_is_total: bool = False,
    num_queries: int = 0,
    assume_M_unit: str = "MiB",
    ratio_field: str = "residual",  # "residual" or "keep_original"
):
    """
    Apply: cost := cost + r_pred, where r_pred = B + k1*eps + k2*M

    Input/Output header: M,epsilon,cost,ratio
    """
    df = pd.read_csv(input_log_path)
    required = {"M", "epsilon", "cost", "ratio"}
    if not required.issubset(df.columns):
        raise ValueError(f"{input_log_path} must have columns {required}")

    M_raw = df["M"].to_numpy(dtype=np.float64)
    eps = df["epsilon"].to_numpy(dtype=np.float64)
    y_hat = df["cost"].to_numpy(dtype=np.float64)

    # M unit handling
    if assume_M_unit == "bytes":
        M_mib = M_raw / (1024.0 * 1024.0)
    else:
        M_mib = M_raw

    # normalize to per-query if needed
    y_hat = base.normalize_to_per_query(y_hat, is_total=estimated_is_total, num_queries=num_queries)

    r_pred = predict_linear_residual(eps, M_mib, coef)
    if residual_clip is not None:
        r_pred = np.clip(r_pred, residual_clip[0], residual_clip[1])

    y_rev = y_hat + r_pred

    # write back in same scale as input
    if estimated_is_total:
        y_rev_out = y_rev * float(num_queries)
    else:
        y_rev_out = y_rev

    if ratio_field == "keep_original":
        ratio_out = df["ratio"]
    else:
        ratio_out = r_pred

    out = pd.DataFrame({
        "M": df["M"],
        "epsilon": df["epsilon"],
        "cost": y_rev_out,
        "ratio": ratio_out,
    })

    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    out.to_csv(output_log_path, index=False, float_format="%.10g")
    print(f"[OK] wrote revised log: {output_log_path}")


def evaluate_holdout_memory_budget_linear(
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
    real_is_total: bool = False,
    num_queries: int = 0,
):
    """
    Hold-out validation (e.g., M=30MiB not used in fitting):
    Compare before/after MAE, RMSE, mean|rel|.
    """
    real_df = base.load_real_avg_ios_per_query(real_csv_path, max_eps=max_eps)
    eps = real_df["epsilon"].to_numpy(dtype=np.float64)
    y_real = real_df["avg_IOs"].to_numpy(dtype=np.float64)
    y_real = base.normalize_to_per_query(y_real, is_total=real_is_total, num_queries=num_queries)

    y_hat = base.compute_model_costs_per_query(
        M_holdout_mib, eps,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy=fetch_strategy,
        mode=mode,
    )

    M_vec = np.full_like(eps, float(M_holdout_mib), dtype=np.float64)
    r_pred = predict_linear_residual(eps, M_vec, coef)
    y_corr = y_hat + r_pred

    err_before = y_real - y_hat
    err_after = y_real - y_corr

    mae_before = float(np.mean(np.abs(err_before)))
    mae_after  = float(np.mean(np.abs(err_after)))
    rmse_before = float(np.sqrt(np.mean(err_before**2)))
    rmse_after  = float(np.sqrt(np.mean(err_after**2)))

    denom = np.maximum(y_real, 1e-12)
    rel_before = err_before / denom
    rel_after  = err_after / denom

    mean_abs_rel_before = float(np.mean(np.abs(rel_before)))
    mean_abs_rel_after  = float(np.mean(np.abs(rel_after)))

    max_abs_rel_before = float(np.max(np.abs(rel_before)))
    max_abs_rel_after  = float(np.max(np.abs(rel_after)))

    print("\n========== Hold-out Validation (linear residual) ==========")
    print(f"M_holdout = {M_holdout_mib} MiB, file = {real_csv_path}")
    print(f"coef [B,k1,k2] = {np.asarray(coef)}")
    print(f"MAE   : before={mae_before:.6g}, after={mae_after:.6g}")
    print(f"RMSE  : before={rmse_before:.6g}, after={rmse_after:.6g}")
    print(f"Mean|rel|: before={mean_abs_rel_before:.6g}, after={mean_abs_rel_after:.6g}")
    print(f"Max |rel|: before={max_abs_rel_before:.6g}, after={max_abs_rel_after:.6g}")


# ============================================================
# Demo runner consistent with your original style
# ============================================================

def run_books10m_linear_calibration_and_revision():
    n = int(1e7)
    seg_size = 16
    ipp = 512
    ps = 4096
    type = "sample"
    data_file = "books_10M_uint64_unique"
    query_file = "books_10M_uint64_unique.query.bin"

    Ms = [10, 20, 40, 60]
    real_csv_files = [
        os.path.join(base.REAL_DIRECTORY, "books_10M_M10_falcon.csv"),
        os.path.join(base.REAL_DIRECTORY, "books_10M_M20_falcon.csv"),
        os.path.join(base.REAL_DIRECTORY, "books_10M_M40_falcon.csv"),
        os.path.join(base.REAL_DIRECTORY, "books_10M_M60_falcon.csv"),
    ]

    # keep consistent with your current pipeline
    real_is_total = True
    num_queries = int(7e5)

    coef = fit_linear_residual_model_global(
        Ms_mib=Ms,
        real_csv_files=real_csv_files,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy="all_in_once",
        mode="point",
        max_eps=64,
        ridge_lambda=1e-6,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    print("Fitted linear residual coef [B,k1,k2]:")
    print(coef)

    # hold-out 30MiB
    holdout_csv = os.path.join(base.REAL_DIRECTORY, "books_10M_M30_falcon.csv")
    evaluate_holdout_memory_budget_linear(
        M_holdout_mib=30,
        real_csv_path=holdout_csv,
        coef=coef,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy="all_in_once",
        mode="point",
        max_eps=64,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    in_log = os.path.join(base.LOG_DIRECTORY, "books_10M_uint64_unique.query.log")
    out_log = os.path.join(base.LOG_DIRECTORY, "books_10M_uint64_unique_linear_revision.query.log")

    apply_linear_residual_to_estimated_log(
        input_log_path=in_log,
        output_log_path=out_log,
        coef=coef,
        residual_clip=None,
        estimated_is_total=False,
        num_queries=0,
        assume_M_unit="MiB",
        ratio_field="residual",
    )


if __name__ == "__main__":
    run_books10m_linear_calibration_and_revision()
