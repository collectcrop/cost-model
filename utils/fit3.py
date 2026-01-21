# residual_reciprocal_fit.py
import os
import numpy as np
import pandas as pd

# ============================================================
# IMPORTANT: set this to your original module filename (without .py)
# Example: if your original file is named "cam_residual.py",
# set BASE_MODULE_NAME = "cam_residual"
# ============================================================
BASE_MODULE_NAME = "fit2"  # TODO: change me
base = __import__(BASE_MODULE_NAME)

# ============================================================
# Residual model:
#   r(eps, M) = B + k1*eps + k2*(1/(eps+eps0)) + k3*M
# (eps0 is a small stabilizer; set eps0=0.0 if you insist on 1/eps)
# ============================================================

def build_residual_features_eps_reciprocal_M(
    eps: np.ndarray,
    M_mib: np.ndarray,
    *,
    eps0: float = 1.0
) -> np.ndarray:
    """
    Design matrix for:
      r = B + k1*eps + k2*inv + k3*M
    where inv = 1/(eps+eps0)

    X = [1, eps, inv, M]
    """
    eps = np.asarray(eps, dtype=np.float64)
    M_mib = np.asarray(M_mib, dtype=np.float64)
    inv = 1.0 / (eps + float(eps0))
    return np.column_stack([np.ones_like(eps), eps, inv, M_mib])


def fit_residual_model_global_eps_reciprocal_M(
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
    ridge_lambda: float = 0.0,
    # align measurement scale
    real_is_total: bool = False,
    num_queries: int = 0,
) -> np.ndarray:
    """
    Fit residual r(eps,M) = y_real - y_hat with:
        r = B + k1*eps + k2/(eps+eps0) + k3*M

    Returns:
        coef = [B, k1, k2, k3]
    """
    if len(Ms_mib) != len(real_csv_files):
        raise ValueError("Ms_mib and real_csv_files must have same length")

    all_X, all_r = [], []

    for M_mib, csv_path in zip(Ms_mib, real_csv_files):
        real_df = base.load_real_avg_ios_per_query(csv_path, max_eps=max_eps)
        eps = real_df["epsilon"].to_numpy(dtype=np.float64)
        y_real = real_df["avg_IOs"].to_numpy(dtype=np.float64)
        y_real = base.normalize_to_per_query(y_real, is_total=real_is_total, num_queries=num_queries)

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
        X = build_residual_features_eps_reciprocal_M(eps, M_vec, eps0=eps0)

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

    return coef  # [B, k1, k2, k3]


def predict_residual_eps_reciprocal_M(
    eps: np.ndarray,
    M_mib: np.ndarray,
    coef: np.ndarray,
    *,
    eps0: float = 1.0
) -> np.ndarray:
    """
    coef = [B, k1, k2, k3]
    r = B + k1*eps + k2/(eps+eps0) + k3*M
    """
    eps = np.asarray(eps, dtype=np.float64)
    M_mib = np.asarray(M_mib, dtype=np.float64)
    X = build_residual_features_eps_reciprocal_M(eps, M_mib, eps0=eps0)
    return X @ np.asarray(coef, dtype=np.float64)


def apply_residual_correction_to_estimated_log_eps_reciprocal_M(
    input_log_path: str,
    output_log_path: str,
    coef: np.ndarray,
    *,
    eps0: float = 1.0,
    residual_clip: tuple | None = None,
    estimated_is_total: bool = False,
    num_queries: int = 0,
    assume_M_unit: str = "MiB",
    ratio_field: str = "residual",  # "residual" or "keep_original"
):
    """
    Apply: cost := cost + r_pred, where r_pred = B + k1*eps + k2/(eps+eps0) + k3*M

    Input/Output header: M,epsilon,cost,ratio
    """
    df = pd.read_csv(input_log_path)
    required = {"M", "epsilon", "cost", "ratio"}
    if not required.issubset(df.columns):
        raise ValueError(f"{input_log_path} must have columns {required}")

    M_raw = df["M"].to_numpy(dtype=np.float64)
    eps = df["epsilon"].to_numpy(dtype=np.float64)
    y_hat = df["cost"].to_numpy(dtype=np.float64)

    if assume_M_unit == "bytes":
        M_mib = M_raw / (1024.0 * 1024.0)
    else:
        M_mib = M_raw

    # normalize to per-query if needed
    y_hat = base.normalize_to_per_query(y_hat, is_total=estimated_is_total, num_queries=num_queries)

    r_pred = predict_residual_eps_reciprocal_M(eps, M_mib, coef, eps0=eps0)
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


def evaluate_holdout_memory_budget_eps_reciprocal_M(
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
    real_is_total: bool = False,
    num_queries: int = 0,
):
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
    r_pred = predict_residual_eps_reciprocal_M(eps, M_vec, coef, eps0=eps0)
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

    print("\n========== Hold-out Validation (eps + 1/eps + M residual) ==========")
    print(f"M_holdout = {M_holdout_mib} MiB, file = {real_csv_path}")
    print(f"coef [B,k1,k2,k3] = {np.asarray(coef)}")
    print(f"MAE   : before={mae_before:.6g}, after={mae_after:.6g}")
    print(f"RMSE  : before={rmse_before:.6g}, after={rmse_after:.6g}")
    print(f"Mean|rel|: before={mean_abs_rel_before:.6g}, after={mean_abs_rel_after:.6g}")
    print(f"Max |rel|: before={max_abs_rel_before:.6g}, after={max_abs_rel_after:.6g}")


# ============================================================
# Demo runner (same style/assumptions as your base script)
# ============================================================

def run_books10m_reciprocal_linear_calibration_and_revision():
    n = int(1e7)
    seg_size = 16
    ipp = 512
    ps = 4096
    type = "sample"
    data_file = "books_10M_uint64_unique"
    query_file = "books_10M_uint64_unique.query.bin"

    Ms = [10, 20, 40, 60]
    real_csv_files = [
        os.path.join(base.REAL_DIRECTORY, "books_10M_M10_falcon_pred.csv"),
        os.path.join(base.REAL_DIRECTORY, "books_10M_M20_falcon_pred.csv"),
        os.path.join(base.REAL_DIRECTORY, "books_10M_M40_falcon_pred.csv"),
        os.path.join(base.REAL_DIRECTORY, "books_10M_M60_falcon_pred.csv"),
    ]

    # Keep consistent with your current base runner
    real_is_total = True
    num_queries = int(3e5)

    coef = fit_residual_model_global_eps_reciprocal_M(
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
        ridge_lambda=1e-6,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    print("Fitted coef [B, k1, k2, k3]:")
    print(coef)

    # Hold-out 30 MiB
    holdout_csv = os.path.join(base.REAL_DIRECTORY, "books_10M_M30_falcon.csv")
    evaluate_holdout_memory_budget_eps_reciprocal_M(
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
        eps0=0.0,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    in_log = os.path.join(base.LOG_DIRECTORY, "books_10M_uint64_unique.query.log")
    out_log = os.path.join(base.LOG_DIRECTORY, "books_10M_uint64_unique_revision.query.log")

    apply_residual_correction_to_estimated_log_eps_reciprocal_M(
        input_log_path=in_log,
        output_log_path=out_log,
        coef=coef,
        eps0=0.0,
        residual_clip=None,
        estimated_is_total=False,
        num_queries=0,
        assume_M_unit="MiB",
        ratio_field="residual",
    )

def run_fb10m_reciprocal_linear_calibration_and_revision():
    n = int(1e7)
    seg_size = 16
    ipp = 512
    ps = 4096
    type = "sample"
    data_file = "fb_10M_uint64_unique"
    query_file = "fb_10M_uint64_unique.range.bin"

    Ms = [10, 20, 40, 60]
    real_csv_files = [
        os.path.join(base.REAL_DIRECTORY, "range_fb_10M_M10_pred.csv"),
        os.path.join(base.REAL_DIRECTORY, "range_fb_10M_M20_pred.csv"),
        os.path.join(base.REAL_DIRECTORY, "range_fb_10M_M40_pred.csv"),
        os.path.join(base.REAL_DIRECTORY, "range_fb_10M_M60_pred.csv"),
    ]

    # Keep consistent with your current base runner
    real_is_total = True
    num_queries = int(3e5)

    coef = fit_residual_model_global_eps_reciprocal_M(
        Ms_mib=Ms,
        real_csv_files=real_csv_files,
        n=n, seg_size=seg_size, ipp=ipp, ps=ps,
        type=type,
        data_file=data_file,
        query_file=query_file,
        fetch_strategy="all_in_once",
        mode="range",
        max_eps=64,
        eps0=0.0,            
        ridge_lambda=1e-6,
        real_is_total=real_is_total,
        num_queries=num_queries,
    )

    print("Fitted coef [B, k1, k2, k3]:")
    print(coef)

    # Hold-out 30 MiB
    # holdout_csv = os.path.join(base.REAL_DIRECTORY, "books_10M_M30_falcon.csv")
    # evaluate_holdout_memory_budget_eps_reciprocal_M(
    #     M_holdout_mib=30,
    #     real_csv_path=holdout_csv,
    #     coef=coef,
    #     n=n, seg_size=seg_size, ipp=ipp, ps=ps,
    #     type=type,
    #     data_file=data_file,
    #     query_file=query_file,
    #     fetch_strategy="all_in_once",
    #     mode="point",
    #     max_eps=64,
    #     eps0=0.0,
    #     real_is_total=real_is_total,
    #     num_queries=num_queries,
    # )

    in_log = os.path.join(base.LOG_DIRECTORY, "fb_10M_uint64_unique.range.log")
    out_log = os.path.join(base.LOG_DIRECTORY, "fb_10M_uint64_unique_revision.range.log")

    apply_residual_correction_to_estimated_log_eps_reciprocal_M(
        input_log_path=in_log,
        output_log_path=out_log,
        coef=coef,
        eps0=0.0,
        residual_clip=None,
        estimated_is_total=False,
        num_queries=0,
        assume_M_unit="MiB",
        ratio_field="residual",
    )
if __name__ == "__main__":
    run_fb10m_reciprocal_linear_calibration_and_revision()
