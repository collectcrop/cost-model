import numpy as np
import scipy
import math
import pandas as pd
from optimalEpsilon import cost_function,expected_DAC,range_cost_function

DATASETS_DIRECTORY = "/mnt/home/zwshi/Datasets/SOSD/"
LOG_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/visualize/data/log/"
REAL_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/include/FALCON/results/log/"
def collect_model_points_for_fixed_M(
    M_bytes,
    epsilon_list,
    n,
    seg_size,
    ipp,
    ps,
    type,
    data_file,
    query_file,
    s="one_by_one",
    mode="point"
):
    """
    对固定 M，在一串 epsilon_list 上调用一次 cost_function，
    得到 model 的 cost 和 h（每个 epsilon 一次 Che）。
    """
    data_path = f"{DATASETS_DIRECTORY}{data_file}"
    query_path = f"{DATASETS_DIRECTORY}{query_file}"

    eps_arr = np.array(epsilon_list, dtype=np.float64)
    cost_model_list = []
    h_model_list = []

    for eps in eps_arr:
        if mode == "point":
            cost_model, h_model = cost_function(
                eps,
                n,
                seg_size,
                M_bytes,
                ipp,
                ps,
                type=type,
                query_file=query_path,
                data_file=data_path,
                s=s,
            )
        elif mode == "range":
            cost_model, h_model = range_cost_function(
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
        
        cost_model_list.append(cost_model)
        h_model_list.append(h_model)

    return eps_arr, np.array(cost_model_list), np.array(h_model_list)
def load_data(file_path,max_eps=64,num_queries=1000000):
    df = pd.read_csv(file_path)
    grouped = df.groupby('epsilon', as_index=False)['avg_IOs'].mean()
    # grouped = grouped.rename(columns={'hit_ratio': 'real'})
    grouped["hit_rates"] = df['hit_ratio']
    grouped = grouped[grouped["epsilon"]<=max_eps] 
    return grouped

def fit_miss_ratio_correction(
    epsilon_list,
    hit_rates_list,
    M_bytes,
    n,
    seg_size,
    ipp,
    ps,
    type,
    data_file,
    query_file,
    s="all_in_once",
    mode="point"
):
    """
    用固定 M + 一串 epsilon 下的 (model hit, real hit) 拟合一个“偏移量”：
        h_real ≈ h_model + B

    返回 A, B，其中 A 固定为 1.0，只修正 B。
    """
    # 1) 收集 model 的 cost, h
    eps_arr, cost_model_arr, h_model_arr = collect_model_points_for_fixed_M(
        M_bytes,
        epsilon_list,
        n,
        seg_size,
        ipp,
        ps,
        type,
        data_file,
        query_file,
        s=s,
        mode=mode,
    )

    # 转成 numpy 数组以防是 pandas Series
    h_model_arr = np.asarray(h_model_arr, dtype=np.float64)
    hit_rates_arr = np.asarray(hit_rates_list, dtype=np.float64)

    # 安全检查：长度需要一致
    if h_model_arr.shape[0] != hit_rates_arr.shape[0]:
        raise ValueError(
            f"Length mismatch: model {h_model_arr.shape[0]} vs real {hit_rates_arr.shape[0]}"
        )

    # 只拟合偏移：B = mean(h_real - h_model)
    residual = hit_rates_arr - h_model_arr
    B = float(residual.mean())

    # A 固定为 1.0
    A = 1.0
    return A, B
def demo_fit():
    
    n = int(1e7)
    seg_size = 16
    ipp = 512
    ps = 4096
    type = "sample"
    data_file = f"books_10M_uint64_unique"
    query_file = f"books_10M_uint64_unique.query.bin"
    M_MiB = 60
    M_bytes = M_MiB * 1024 * 1024
    
    fn = f"books_10M_M{M_MiB}_falcon.csv"
    df = load_data(REAL_DIRECTORY+fn)
    hit_rates_list = df["hit_rates"]
    epsilon_list = df["epsilon"].tolist()
    
    A, B = fit_miss_ratio_correction(
        epsilon_list,
        hit_rates_list,
        M_bytes,
        n,
        seg_size,
        ipp,
        ps,
        type,
        data_file,
        query_file,
    )

    print("Fitted A:", A)
    print("Fitted B:", B)

def demo_fit_range():
    n = int(1e7)
    seg_size = 16
    ipp = 512
    ps = 4096
    type = "sample"  # 对 range_cost_function 实际无用，但接口保留
    data_file  = "fb_10M_uint64_unique"
    query_file = "range_query_fb_uu.bin"  # 按你真实文件名改
    M_MiB = 60
    M_bytes = M_MiB * 1024 * 1024

    fn = f"range_fb_10M_M{M_MiB}.csv"  # 你的 range 结果 CSV
    df = load_data(REAL_DIRECTORY + fn)
    epsilon_list   = df["epsilon"].to_numpy()
    hit_rates_list = df["hit_rates"].to_numpy()

    A, B = fit_miss_ratio_correction(
        epsilon_list,
        hit_rates_list,
        M_bytes,
        n,
        seg_size,
        ipp,
        ps,
        type,
        data_file,
        query_file,
        s="all_in_once",
        mode="range",
    )

    print("[RANGE] Fitted A:", A)
    print("[RANGE] Fitted B:", B)
    
if __name__ == "__main__":
    demo_fit_range()