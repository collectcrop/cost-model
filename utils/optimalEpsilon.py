import math
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy.optimize import brentq  # 用于解非线性方程
from scipy.special import zeta     # Riemann zeta 函数
from collections import Counter
from scipy.signal import fftconvolve

alpha = 1
DATASETS_DIRECTORY = "/mnt/home/zwshi/Datasets/SOSD/"
LOG_DIRECTORY = "/mnt/home/zwshi/learned-index/cost-model/visualize/data/log/"
# 一阶修正：有效缓存容量参数
A_DEFAULT = 1.0   
B_DEFAULT = 0.0  

def build_uniform_box_kernel(epsilon):
    """离散均匀 box kernel 长度 L=2*epsilon+1,已归一化"""
    L = 2 * epsilon + 1
    return np.ones(L, dtype=np.float64) / L

def triangular_kernel_from_box(epsilon):
    """如果 g 和 h 都是长度 L 的均匀 box, 则 k = g*h 是三角核(长度 2L-1)"""
    L = 2 * epsilon + 1
    # discrete triangular: [1,2,3,...,L-1,L,L-1,...,1] normalized by L^2
    up = np.arange(1, L+1, dtype=np.float64)
    down = up[-2::-1]  # L-1 down to 1
    tri = np.concatenate([up, down])
    return tri / (L * L)  # normalization: sum(tri) = L^2 / L^2 = 1

def estimate_page_counts_from_queryfile(query_file, data, epsilon, ipp, use_fft=False):
    """
    args:
      query_file: binary file path or numpy array of query keys (uint64)
      data: sorted data keys (np.array dtype uint64)
      epsilon: int
      ipp: items per page
    returns:
      page_counts: np.array length num_pages, expected counts per page (sum ~= Q * (2eps+1))
      T_pos: np.array length N, expected counts per position
      Q: queries length
    """
    # 读取 queries 支持文件或直接数组
    
    if isinstance(query_file, str):
        queries = np.fromfile(query_file, dtype=np.uint64)
    else:
        queries = np.asarray(query_file, dtype=np.uint64)

    Q = len(queries)

    assert isinstance(data, np.ndarray)
    N = len(data)
    # pos = index of largest data <= query -> searchsorted -1
    pos = np.searchsorted(data, queries, side='right') - 1
    pos = np.clip(pos, 0, N-1).astype(np.int64)
    

    # 1)  construct H(p) (len=N)
    H = np.bincount(pos, minlength=N).astype(np.float64)  # sum(H) == Q

    # 2) construct k = g * h. assume g = uniform box, h = uniform box => k = triangular
    k = triangular_kernel_from_box(epsilon)  # length K = 4*eps + 1, sums to 1

    # 3) 卷积 H * k -> T positions (期望位置访问次数). 使用 'same' 保持长度 N
    if use_fft:
        # 对于非常大 N 可考虑 FFT 卷积实现，如 scipy.signal.fftconvolve
        
        T = fftconvolve(H, k, mode='same')
    else:
        T = np.convolve(H, k, mode='same')

    # 注：每个 query 平均导致 L=2ε+1 个位置被访问 -> sum(T) ~= Q * L 。
    # 如果需要保证 sum == Q*L, 检查 kernel sum == 1, H sum == Q -> sum(T) == Q

    # 4) 按页聚合
    num_pages = math.ceil(N / ipp)
    pad_len = num_pages * ipp - N
    if pad_len > 0:
        T_padded = np.concatenate([T, np.zeros(pad_len, dtype=T.dtype)])
    else:
        T_padded = T
    page_counts = T_padded.reshape(num_pages, ipp).sum(axis=1)  # expected counts per page

    return page_counts, T, Q

def estimate_page_counts_from_range_queryfile(lo_keys, hi_keys, data, epsilon, ipp, use_fft=False):
    # map lo->positions then to pages
    N = len(data)
    lo_pos = np.searchsorted(data, lo_keys, side='right') - 1
    hi_pos = np.searchsorted(data, hi_keys, side='right') - 1
    lo_pos = np.clip(lo_pos, 0, N-1)
    hi_pos = np.clip(hi_pos, 0, N-1)
    # page indices
    lo_pages = lo_pos // ipp
    hi_pages = hi_pos // ipp
    num_pages = math.ceil(N / ipp)
    count = Counter()

    for i in range(len(lo_pages)):
        for page in range(lo_pages[i],hi_pages[i]+1):
            count[page] += 1
            
    for i in range(len(lo_pages)):
        start_pos = max(0,lo_pos[i]-int(epsilon))
        for page in range(start_pos//ipp,lo_pages[i]):
            page_start = page * ipp
            page_end   = min((page + 1) * ipp - 1, N - 1)
            count[page] += (page_end - max(page_start,start_pos))/(2*epsilon+1)
    
    return count                  #, k_pages, offset_min


def extract_data_gap_distribution(data_file):
    """
    input:
        data_file: binary file path of sorted data keys (uint64)
    output:
        miu: float, mean of gaps
        sigma: float, stddev of gaps
    """
    data = np.fromfile(data_file, dtype=np.uint64)[1:]
    gaps = np.diff(data)
    miu = np.mean(gaps)
    sigma = np.std(gaps)
    return miu, sigma

def extract_query_distribution(filename,data,epsilon,ipp):
    """
    input:
        filename: query filename (binary file with uint64 keys)
    output:
        probs: np.array, where probs[i] = q(i), i.e., probability of the i-th most popular key
    """
    queries = np.fromfile(filename, dtype=np.uint64)
    pos = np.searchsorted(data, queries, side='right') - 1
    total_keys = len(data)
    # num_pages = int(np.ceil(total_keys / ipp))
    count = Counter()
    for p in pos:
        start_pos = max(0, p - epsilon)
        end_pos   = min(total_keys - 1, p + epsilon)
        total_len = end_pos - start_pos + 1
        start = start_pos//ipp
        end = end_pos//ipp
        for page in range(start,end+1):
            count[page] += 1
            # page_start = page * ipp
            # page_end   = min((page + 1) * ipp - 1, total_keys - 1)
            # overlap    = max(0, min(end_pos, page_end) - max(start_pos, page_start) + 1)
            # if overlap > 0:
            #     # add weight 1
            #     count[page] += overlap / total_len
    # print(count)        
    # count = Counter(queries)
    # total = len(queries)
    total = sum(count.values())
    sorted_freqs = sorted(count.values(), reverse=True)
    probs = np.array([f / total for f in sorted_freqs], dtype=np.float64)
    return probs

def zipf_popularity(N, alpha):
    norm_const = sum(1 / (i ** alpha) for i in range(1, N + 1))
    return np.array([1 / (i ** alpha) / norm_const for i in range(1, N + 1)])

# def che_characteristic_time(qs, C, Q):
#     # qs: array of popularity q(i)
#     print("[*] starting solve characteristic time")
#     m = int(np.sum(qs > 0))
#     def f(t):
#         return np.sum(1 - np.exp(-qs * t)) - C
#     # root-finding to solve C = Σ(1 - e^{-q_i t})
#     print(C,m)
#     return brentq(f, 1e-6, 1e6)

def che_characteristic_time(qs, C, t0=1e-9, grow=10.0, max_iter=60):
    """
    Solve for t_C in:  C = sum_i (1 - exp(-q_i * t_C))
    - qs: array-like of per-object request rates or counts (q_i >= 0)
    - C : cache capacity measured in "number of objects" (or equivalently the unit that matches 1 per object)
    - Q : (optional) total queries in the window; not used in solving, here just for signature compatibility

    Returns:
        t_C (float): finite positive solution; 0.0 when C<=0; np.inf when C >= m (no finite solution).
    """
    qs = np.asarray(qs, dtype=float)
    if np.any(~np.isfinite(qs)) or np.any(qs < 0):
        raise ValueError("qs must be finite and >= 0")
    m = int(np.sum(qs > 0))

    # Boundary/degenerate cases
    if C <= 0:
        return 0.0
    if C >= m:
        # No finite solution; in practice "all objects fit" => t_C = +inf
        return np.inf

    # Scale to improve conditioning
    qpos = qs[qs > 0]
    qbar = float(np.mean(qpos)) if qpos.size > 0 else 1.0
    r = np.where(qs > 0, qs / qbar, 0.0)

    def f_tau(tau):
        return np.sum(1.0 - np.exp(-r * tau)) - C

    # Bracket automatically on (t0, +inf)
    a, b = t0, t0
    fa = f_tau(a)
    for _ in range(max_iter):
        b *= grow
        fb = f_tau(b)
        if fa * fb <= 0:
            break
    else:
        # Extremely pathological scaling; fall back to a wide static bracket
        a, b = 1e-12, 1e12

    tau = brentq(f_tau, a, b)
    return tau / qbar


def che_hit_rates(qs, t_C):
    return 1 - np.exp(-qs * t_C)

def predict_height_segments(epsilon,epsilon_i,n,k):
    h = 1
    num = math.ceil(n/(k*epsilon**2))
    segments = num
    while (num>1):
        h = h+1
        num = math.ceil(num/(k*epsilon_i**2))
        segments += num
    return h,segments            # 1+math.ceil(math.log(n/(k*epsilon**2),k*epsilon_i**2))


# def expected_DAC(epsilon, ipp):
#     dac = 0
#     for k in range(ipp + 1):
#         term = 1 + math.ceil((epsilon - k) / ipp) + math.ceil((epsilon - ipp + k) / ipp)
#         dac += term
#     return dac / ipp

def expected_DAC(epsilon, ipp ,s="all_in_once"):
    if s == "all_in_once":
        return 1 + (2*epsilon/ipp)
    elif s == "one_by_one":
        return 1 + (epsilon/ipp)

def expected_IAC(epsilon, ipp):
    return 1 + (2*epsilon/ipp)

def uniform_ratio(C,N):
    return C / N

def zipf_ratio(C,N,alpha):
    qs = zipf_popularity(N, alpha)
    t_C = che_characteristic_time(qs, C)
    hit_rates = che_hit_rates(qs, t_C)
    return np.sum(qs * hit_rates)

# def sample_ratio(C,N,qs,Q):
#     t_C = che_characteristic_time(qs, C, Q)
#     print("[+] successfully solved characteristic_time")
#     hit_rates = che_hit_rates(qs, t_C)
#     return np.sum(qs * hit_rates)

def sample_ratio(C, N, qs, Q=0):
    """
    Return "cache hit ratio" over Q queries.
    - If t_C is finite: hits = sum_i q_i * h_i with h_i = 1 - exp(-q_i * t_C)
    - If t_C is +inf (C >= m): use h = max(0, (Q - m)/Q) => total hits = h * Q = max(0, Q - m)
    """
    t_C = che_characteristic_time(qs, C)
    print("[+] successfully solved characteristic_time:", t_C)
    m = int(np.sum(np.asarray(qs) > 0))
    if np.isinf(t_C):
        # All objects fit. Under the "first-time miss only" counting over a length-Q trace:
        # hit_rates = max(0, Q - m)/Q
        hit_rates = 1
        return hit_rates

    # Otherwise, use standard Che hit rates)
    hit_rates = che_hit_rates(qs, t_C)
    return float(np.sum(qs * hit_rates))
def validate_ratio(ratio):
    if ratio >= 1.0:
        h = 1.0
    elif ratio <= 0:
        h = 0.0
    else:
        h = ratio
    return h

def model_cost_given_capacity(
    epsilon,
    n,
    seg_size,
    ipp,
    ps,
    C_pages,
    type="sample",
    data_file="",
    query_file="",
    s="all_in_once",
):
    """
    在给定“页数容量 C_pages”时，调用现有 cost_function 得到模型 cost。
    注意：这里强行构造一个 M 使得 CAM 里看到的 C 就是 C_pages。
    """
    # 对应索引大小
    M_index = n * seg_size / (2 * epsilon)

    # buffer 部分 = C_pages * ps
    M_buffer = C_pages * ps

    # 总内存 M_eff = index + buffer
    M_eff = M_index + M_buffer

    # 用原来的 cost_function 计算 cost（此时 cost_function 里算出的 C 就是 C_pages）
    cost, h = cost_function(
        epsilon,
        n,
        seg_size,
        M_eff,
        ipp,
        ps,
        type=type,
        query_file=query_file,
        data_file=data_file,
        s=s,
    )
    return cost


# def join_cost_function(
#     epsilon, n, seg_size, M, ipp, ps,
#     data_file="", join_file="", par_file="", bitmap_file="",
#     s_point="all_in_once",          # 与 cost_function 一致：expected_DAC(eps, ipp, s)
#     A=A_DEFAULT, B=B_DEFAULT,
#     type="sample",                 # 目前 join 基本都应走 sample
#     return_detail=False,
# ):
#     # -----------------------
#     # 0) 读取数据与 join keys
#     # -----------------------
#     data = np.fromfile(data_file, dtype=np.uint64)[1:]
#     join_keys = np.fromfile(join_file, dtype=np.uint64)
#     Q_total = int(len(join_keys))
#     if Q_total == 0:
#         return (0.0, 0.0, {"Q_total": 0}) if return_detail else (0.0, 0.0)

#     # join_partition 期望 join_keys 单调；不单调则排序（保持安全）
#     if np.any(join_keys[1:] < join_keys[:-1]):
#         join_keys = np.sort(join_keys)

#     # -----------------------
#     # 1) 读取 partition 信息
#     # -----------------------
#     if (not os.path.exists(par_file)) or (not os.path.exists(bitmap_file)):
#         raise FileNotFoundError(
#             f"partition files not found: {par_file} / {bitmap_file}. "
#             f"Please run generate_query.join_partition(...) for this epsilon first."
#         )

#     lengths = np.fromfile(par_file, dtype=np.int64)
#     bitmap  = np.fromfile(bitmap_file, dtype=np.int8)

#     if lengths.size != bitmap.size:
#         raise ValueError(f"lengths.size({lengths.size}) != bitmap.size({bitmap.size})")

#     if int(lengths.sum()) != Q_total:
#         raise ValueError(f"sum(lengths)={int(lengths.sum())} != Q_total={Q_total}")

#     # -----------------------
#     # 2) 切分为 point_keys 与 range_pairs
#     # -----------------------
#     point_list = []
#     range_lo = []
#     range_hi = []

#     off = 0
#     for L, b in zip(lengths, bitmap):
#         L = int(L)
#         seg = join_keys[off:off+L]
#         if b == 0:
#             point_list.append(seg)
#         else:
#             # 该 partition 用一次 range scan 覆盖
#             range_lo.append(int(seg[0]))
#             range_hi.append(int(seg[-1]))
#         off += L

#     point_keys = np.concatenate(point_list) if len(point_list) else np.array([], dtype=np.uint64)
#     range_lo = np.array(range_lo, dtype=np.uint64)
#     range_hi = np.array(range_hi, dtype=np.uint64)

#     Q_point = int(len(point_keys))
#     P_range = int(len(range_lo))   # range partition 数（每个对应一次 range scan）

#     # -----------------------
#     # 3) 计算 cache 容量（页）与总页数
#     # -----------------------
#     M_index  = n * seg_size / (2 * epsilon)
#     M_buffer = M - M_index
#     C = M_buffer / ps
#     total_pages = math.ceil(n / ipp)

#     # C<=0：无 buffer，hit=0
#     if C <= 0:
#         h = 0.0
#         # 直接返回“无 cache”下的摊销 I/O
#         miss_point = expected_DAC(epsilon, ipp, s_point) * Q_point
#         if P_range > 0:
#             pos_lo = np.searchsorted(data, range_lo, side='right') - 1
#             pos_hi = np.searchsorted(data, range_hi, side='right') - 1
#             RDAC = pos_hi/ipp - pos_lo/ipp + 1 + 2*epsilon/ipp
#             miss_range = float(np.sum(RDAC))
#         else:
#             miss_range = 0.0
#         cost = (miss_point + miss_range) / Q_total
#         detail = {"Q_total": Q_total, "Q_point": Q_point, "P_range": P_range, "C_pages": float(C)}
#         return (cost, h, detail) if return_detail else (cost, h)

#     # -----------------------
#     # 4) 合并 point/range 的 page access 分布 => h
#     # -----------------------
#     # 4.1 point: 复用 point 的 estimate_page_counts
#     if Q_point > 0:
#         pc_point, _, _ = estimate_page_counts_from_queryfile(point_keys, data, epsilon, ipp)
#     else:
#         pc_point = np.zeros(math.ceil(len(data) / ipp), dtype=np.float64)

#     # 4.2 range: 复用 range 的 estimate_page_counts
#     if P_range > 0:
#         pc_range_counter = estimate_page_counts_from_range_queryfile(range_lo, range_hi, data, epsilon, ipp)
#         pc_range = np.zeros_like(pc_point, dtype=np.float64)
#         for p, c in pc_range_counter.items():
#             if 0 <= p < pc_range.size:
#                 pc_range[p] += c
#     else:
#         pc_range = np.zeros_like(pc_point, dtype=np.float64)

#     pc_mix = pc_point + pc_range
#     total_req = float(pc_mix.sum())
#     if total_req <= 0:
#         h = 0.0
#     else:
#         q = pc_mix / total_req
#         q = np.sort(q)[::-1]
#         buffer_ratio = sample_ratio(C, total_pages, q, Q=int(total_req))
#         h = validate_ratio(buffer_ratio)
#         h = validate_ratio(A * h + B)

#     # -----------------------
#     # 5) 计算 join 的 avg miss I/Os（按 join key 摊销）
#     # -----------------------
#     miss_point = (1.0 - h) * expected_DAC(epsilon, ipp, s_point) * Q_point

#     if P_range > 0:
#         pos_lo = np.searchsorted(data, range_lo, side='right') - 1
#         pos_hi = np.searchsorted(data, range_hi, side='right') - 1
#         RDAC = pos_hi/ipp - pos_lo/ipp + 1 + 2*epsilon/ipp   # 与 range_cost_function 一致
#         miss_range = (1.0 - h) * float(np.sum(RDAC))         # 每个 range partition 执行一次
#     else:
#         RDAC = np.array([], dtype=np.float64)
#         miss_range = 0.0

#     cost = (miss_point + miss_range) / Q_total

#     if not return_detail:
#         return cost, h

#     detail = {
#         "Q_total": Q_total,
#         "Q_point": Q_point,
#         "P_range": P_range,
#         "C_pages": float(C),
#         "total_req_pages_mix": total_req,
#         "miss_point_total": float(miss_point),
#         "miss_range_total": float(miss_range),
#         "RDAC_avg_per_range_part": float(RDAC.mean()) if RDAC.size else 0.0,
#     }
#     return cost, h, detail

def join_cost_function(
    epsilon,
    n,
    seg_size,
    M,
    ipp,
    ps,
    data_file="",
    join_file="",
    par_file="",
    bitmap_file="",
    A=A_DEFAULT,   # 这里不再用于 Che 修正，但保留签名兼容
    B=B_DEFAULT,
    assume_sorted=True,
    return_detail=False,
):
    """
    Sorted-order hit-rate model:
      - misses = #distinct pages touched (union of page intervals)
      - hit rate = (n_ref - N_distinct) / n_ref
      - cost (avg physical IO per join key) = N_distinct / Q

    Partition rule:
      - bitmap=0 (point region): execute as point probes per key with window [pos-eps, pos+eps]
      - bitmap=1 (range region): execute as a range scan covering the whole segment window
    """

    data = np.fromfile(data_file, dtype=np.uint64)[1:]
    queries = np.fromfile(join_file, dtype=np.uint64)
    Q = int(len(queries))
    N = int(len(data))
    if Q == 0 or N == 0:
        if return_detail:
            return 0.0, 0.0, {"Q": Q, "n_refs": 0, "N_distinct": 0}
        return 0.0, 0.0

    # sorted precondition (optional safeguard)
    if assume_sorted and np.any(queries[1:] < queries[:-1]):
        queries = np.sort(queries)

    lengths = np.fromfile(par_file, dtype=np.int64)
    bitmap  = np.fromfile(bitmap_file, dtype=np.int8)
    if lengths.size != bitmap.size:
        raise ValueError(f"lengths.size({lengths.size}) != bitmap.size({bitmap.size})")
    if int(lengths.sum()) != Q:
        raise ValueError(f"sum(lengths)={int(lengths.sum())} != Q={Q}")

    # Cache capacity (pages) — 用于检查 sorted-order “只 miss 一次” 前提是否可能被破坏
    M_index  = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C_pages  = M_buffer / ps
    # 一个实用的 sufficient condition：至少能容纳单次 point-window 的最大页宽
    C_delta  = 1 + int(math.ceil((2.0 * epsilon) / ipp))

    # 辅助：把 interval [l,r] 在线并入 union（intervals 输入应当按 l 非递减）
    union_len = 0
    curL = None
    curR = None
    def push_interval(l, r):
        nonlocal union_len, curL, curR
        if l > r:
            return
        if curL is None:
            curL, curR = l, r
            union_len += (curR - curL + 1)
            return
        if l > curR + 1:
            curL, curR = l, r
            union_len += (curR - curL + 1)
        else:
            if r > curR:
                union_len += (r - curR)
                curR = r

    # 统计逻辑 page references 次数
    n_refs = 0

    off = 0
    for L, b in zip(lengths, bitmap):
        L = int(L)
        seg = queries[off:off+L]
        off += L

        if b == 0:
            # point: per key interval [pos-eps, pos+eps] -> pages [l,r]
            pos = np.searchsorted(data, seg, side="right") - 1
            pos = np.clip(pos, 0, N-1).astype(np.int64)

            start_pos = np.maximum(0, pos - int(epsilon))
            end_pos   = np.minimum(N-1, pos + int(epsilon))
            l_pages   = (start_pos // ipp).astype(np.int64)
            r_pages   = (end_pos   // ipp).astype(np.int64)

            # logical refs add
            n_refs += int(np.sum(r_pages - l_pages + 1))

            # union add (intervals are monotone in sorted order)
            for l, r in zip(l_pages, r_pages):
                push_interval(int(l), int(r))

        else:
            # range: one scan interval for the whole partition
            # use first/last key to bound, then expand by epsilon
            lo = int(seg[0])
            hi = int(seg[-1])

            pos_lo = int(np.searchsorted(data, lo, side="right") - 1)
            pos_hi = int(np.searchsorted(data, hi, side="right") - 1)
            pos_lo = max(0, min(N-1, pos_lo))
            pos_hi = max(0, min(N-1, pos_hi))

            start_pos = max(0, pos_lo - int(epsilon))
            end_pos   = min(N-1, pos_hi + int(epsilon))
            l = start_pos // ipp
            r = end_pos   // ipp

            # logical refs: scan each page once
            n_refs += int(r - l + 1)

            # union add
            push_interval(int(l), int(r))

    if n_refs <= 0:
        if return_detail:
            return 0.0, 0.0, {"Q": Q, "n_refs": 0, "N_distinct": int(union_len)}
        return 0.0, 0.0

    # sorted-order hit estimate
    h = (n_refs - union_len) / float(n_refs)
    if h < 0: h = 0.0
    if h > 1: h = 1.0

    # avg physical IO per join key = distinct pages / Q
    cost = union_len / float(Q)

    # 如果 cache 太小，sorted-order “只 miss 一次” 可能乐观：给出诊断信息
    detail = {
        "Q": Q,
        "n_refs": int(n_refs),
        "N_distinct": int(union_len),
        "h_sorted": float(h),
        "C_pages": float(C_pages),
        "C_delta": int(C_delta),
        "may_be_optimistic": bool(C_pages < C_delta),
    }
    return (cost, h, detail) if return_detail else (cost, h)


# def join_cost_function(
#     epsilon,
#     n,
#     seg_size,
#     M,
#     ipp,
#     ps,
#     data_file="",
#     join_file="",
#     s="all_in_once",
#     A=A_DEFAULT,
#     B=B_DEFAULT,
#     assume_sorted=True,
#     return_detail=True,
# ):
#     """
#     Join probe cost model (all-point, sorted order):
#     - Estimate cache hit rate using Theorem (ordered queries):
#         h = (n_refs - N_distinct) / n_refs
#       where:
#         n_refs     = total page references in the last-mile trace
#         N_distinct = number of distinct pages touched (union of windows)
#     - Return avg page-misses per join key (i.e., expected I/Os per query).
#     """

#     # Resolve paths (consistent with getExpectedCostPerEpsilon usage)
#     data_path = data_file
#     join_path = join_file

#     # Load B.key (sorted data) and join keys
#     data = np.fromfile(data_path, dtype=np.uint64)[1:]  # keep consistent with cost_function :contentReference[oaicite:2]{index=2}
#     queries = np.fromfile(join_path, dtype=np.uint64)

#     Q = int(len(queries))
#     if Q == 0:
#         if return_detail:
#             return 0.0, 0.0, {"Q": 0, "n_refs": 0, "N_distinct": 0}
#         return 0.0, 0.0

#     # Ensure sorted order (theorem requires contiguity per page block)
#     if not assume_sorted:
#         queries = np.sort(queries)
#     else:
#         # If caller claims sorted but it's not, sort defensively
#         if np.any(queries[1:] < queries[:-1]):
#             queries = np.sort(queries)

#     N = len(data)
#     if N == 0:
#         if return_detail:
#             return 0.0, 0.0, {"Q": Q, "n_refs": 0, "N_distinct": 0}
#         return 0.0, 0.0

#     # Cache capacity (in pages), and theorem precondition C >= C_delta
#     M_index = n * seg_size / (2 * epsilon)
#     M_buffer = M - M_index
#     C_pages = M_buffer / ps
#     C_delta = 1 + int(math.ceil((2.0 * epsilon) / ipp))  # max pages per query window (approx)

#     # Map each join key to position in data (B.key)
#     pos = np.searchsorted(data, queries, side="right") - 1  # :contentReference[oaicite:3]{index=3}
#     pos = np.clip(pos, 0, N - 1).astype(np.int64)

#     # Build page intervals per query: [l_i, r_i]
#     start_pos = np.maximum(0, pos - int(epsilon))
#     end_pos = np.minimum(N - 1, pos + int(epsilon))
#     l_pages = (start_pos // ipp).astype(np.int64)
#     r_pages = (end_pos // ipp).astype(np.int64)

#     # Total page references in the trace
#     lens = (r_pages - l_pages + 1)
#     n_refs = int(lens.sum())

#     if n_refs <= 0:
#         if return_detail:
#             return 0.0, 0.0, {"Q": Q, "n_refs": 0, "N_distinct": 0}
#         return 0.0, 0.0

#     # Distinct pages touched = union length of intervals (one pass merge)
#     # Since queries are sorted by key, l_pages/r_pages are non-decreasing in practice.
#     N_distinct = 0
#     cur_l = int(l_pages[0])
#     cur_r = int(r_pages[0])
#     for l, r in zip(l_pages[1:], r_pages[1:]):
#         l = int(l); r = int(r)
#         if l > cur_r + 0:   # disjoint
#             N_distinct += (cur_r - cur_l + 1)
#             cur_l, cur_r = l, r
#         else:               # overlap / touch
#             if r > cur_r:
#                 cur_r = r
#     N_distinct += (cur_r - cur_l + 1)

#     # Ordered-theorem hit rate (exact under C_pages >= C_delta)
#     h = (n_refs - N_distinct) / float(n_refs)
#     h = validate_ratio(h)  # reuse your clamp helper :contentReference[oaicite:4]{index=4}
#     h = A * h + B          # keep your 1st-order correction hook :contentReference[oaicite:5]{index=5}
#     h = validate_ratio(h)

#     # Average page misses per query == N_distinct / Q
#     avg_miss_pages = N_distinct / float(Q)

#     # For compatibility with your existing interface, also report "avg pages touched per query"
#     # This is empirical from trace (n_refs/Q), not the analytic expected_DAC.
#     avg_pages_touched = n_refs / float(Q)

#     detail = {
#         "Q": Q,
#         "C_pages": float(C_pages),
#         "C_delta": int(C_delta),
#         "theorem_applicable": bool(C_pages >= C_delta),
#         "n_refs": int(n_refs),
#         "N_distinct": int(N_distinct),
#         "avg_pages_touched": float(avg_pages_touched),
#         "avg_miss_pages": float(avg_miss_pages),
#         "h": float(h),
#     }

#     if return_detail:
#         return avg_miss_pages, h, detail
#     return avg_miss_pages, h

def range_cost_function(epsilon, n, seg_size, M, ipp, ps, query_file="", data_file="",
                        A=A_DEFAULT,B=B_DEFAULT):
    M_index = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C = M_buffer/ps
    total_pages = math.ceil(n / ipp)
    data = np.fromfile(data_file, dtype=np.uint64)[1:]
    queries = np.fromfile(query_file, dtype=np.uint64).reshape(-1, 2)
    lo_keys,hi_keys = queries[:,0],queries[:,1]
    pos_lo = np.searchsorted(data, lo_keys, side='right') - 1
    pos_hi = np.searchsorted(data, hi_keys, side='right') - 1
    # keys = pos_hi - pos_lo
    # RDAC = np.ceil(keys/ipp) + np.floor(1-(pos_hi%ipp-pos_lo%ipp)/ipp) + 2*epsilon/ipp
    RDAC = pos_hi/ipp - pos_lo/ipp + 1 + 2*epsilon/ipp
    page_counts = estimate_page_counts_from_range_queryfile(lo_keys, hi_keys, data, epsilon, ipp)
    total = sum(page_counts.values())
    q = np.array([f / total for f in page_counts.values()], dtype=np.float64)
    q = np.sort(q)[::-1]
    
    buffer_ratio = sample_ratio(C, total_pages, q)
    # print(buffer_ratio)
    h = validate_ratio(buffer_ratio)
    h = A * h + B
    print((1-h)*RDAC.sum()/len(queries))
    return (1-h)*RDAC.sum()/len(queries), h
    
def cost_function(epsilon, n, seg_size, M, ipp, ps, type="uniform", query_file="", data_file="",s="all_in_once",
                  A=A_DEFAULT,B=B_DEFAULT):
    M_index = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C = M_buffer/ps
    total_pages = math.ceil(n / ipp)
    if (C==0.0):
        h = 0.0
    else:
        if type == "uniform":
            buffer_ratio = uniform_ratio(C, total_pages)
        elif type == "sample":
            data = np.fromfile(data_file,dtype=np.uint64)[1:]
            # q = extract_query_distribution(query_file,data,epsilon,ipp)
            page_counts, Tpos, Q = estimate_page_counts_from_queryfile(query_file, data, epsilon, ipp)
            total_page_requests = page_counts.sum()
            q = page_counts / total_page_requests
            q = np.sort(q)[::-1]
            buffer_ratio = sample_ratio(C, total_pages, q, Q)
        elif type == "zipf":
            buffer_ratio = zipf_ratio(C, total_pages,alpha)
        
        h = validate_ratio(buffer_ratio)
        h = A * h + B
        
    return (1 - h) * expected_DAC(epsilon, ipp, s), h


def getExpectedJoinCostPerEpsilon(ipp, seg_size, M, n, ps,data_file="",query_file="",
                                   A=A_DEFAULT,B=B_DEFAULT):
    data = f"{DATASETS_DIRECTORY}{data_file}"
    query = f"{DATASETS_DIRECTORY}{query_file}"
    par = f"{DATASETS_DIRECTORY}{query_file}".replace(".bin",".par")
    bitmap = f"{DATASETS_DIRECTORY}{query_file}".replace(".bin",".bitmap")
    eps_list = []
    cost_list = []
    h_list = []
    time_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    for eps in range(least_eps if (least_eps%2==0) else least_eps+1, 65, 2):
        t1 = time.time()
        cost,h = join_cost_function(eps, n, seg_size, M, ipp, ps, data, query, par, bitmap, A=A,B=B)
        eps_list.append(eps)
        cost_list.append(cost)
        h_list.append(h)
        t2 = time.time()
        print(f"eps: {eps}, cost: {cost}, ratio: {h}, time: {t2-t1}")
        time_list.append(t2-t1)
    print(eps_list)
    print("cost:",cost_list)
    print("ratio:",h_list)
    
    print("group avg time:", sum(time_list)/len(time_list))
    
    log_filename = f"{query_file}.join.log".replace(".bin","")
    with open(LOG_DIRECTORY+log_filename,'a') as f:
        f.write("M,epsilon,cost,ratio\n")
        for i in range(len(cost_list)):
            f.write(f"{M>>20},{eps_list[i]},{cost_list[i]},{h_list[i]}\n")
    return eps_list, cost_list
def getExpectedRangeCostPerEpsilon(ipp, seg_size, M, n, ps,data_file="",query_file="",
                                   A=A_DEFAULT,B=B_DEFAULT):
    data = f"{DATASETS_DIRECTORY}{data_file}"
    query = f"{DATASETS_DIRECTORY}{query_file}"
    eps_list = []
    cost_list = []
    h_list = []
    time_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    for eps in range(least_eps, 65, 2):
        t1 = time.time()
        cost,h = range_cost_function(eps, n, seg_size, M, ipp, ps, query, data,A=A,B=B)
        eps_list.append(eps)
        cost_list.append(cost)
        h_list.append(h)
        t2 = time.time()
        print(f"eps: {eps}, cost: {cost}, ratio: {h}, time: {t2-t1}")
        time_list.append(t2-t1)
    print(eps_list)
    print("cost:",cost_list)
    print("ratio:",h_list)
    
    print("group avg time:", sum(time_list)/len(time_list))
    
    log_filename = f"{query_file}.range.log".replace(".bin","")
    with open(LOG_DIRECTORY+log_filename,'a') as f:
        f.write("M,epsilon,cost,ratio\n")
        for i in range(len(cost_list)):
            f.write(f"{M>>20},{eps_list[i]},{cost_list[i]},{h_list[i]}\n")
    return eps_list, cost_list

def getExpectedCostPerEpsilon(ipp, seg_size, M, n, ps,type="uniform",data_file="",query_file="",s="all_in_once",
                              A=A_DEFAULT,B=B_DEFAULT):
    data = f"{DATASETS_DIRECTORY}{data_file}"
    query = f"{DATASETS_DIRECTORY}{query_file}"
    eps_list = []
    cost_list = []
    h_list = []
    time_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    for eps in range(least_eps if (least_eps%2==0) else least_eps+1, 129, 2):
        t1 = time.time()
        cost,h = cost_function(eps, n, seg_size, M, ipp, ps, type, query, data, s, A=A,B=B)
        eps_list.append(eps)
        cost_list.append(cost)
        h_list.append(h)
        t2 = time.time()
        print(f"eps: {eps}, cost: {cost}, ratio: {h}, time: {t2-t1}")
        time_list.append(t2-t1)
    print(eps_list)
    print("cost:",cost_list)
    print("ratio:",h_list)
    groups = {
        "8-16":   (8, 16),
        "16-32":  (16, 32),
        "32-64":  (32, 64),
        "64-128": (64, 128),
    }

    group_times = {name: [] for name in groups}

    for eps, t in zip(eps_list, time_list):
        for name, (lo, hi) in groups.items():
            # 左闭右开区间：[lo, hi)
            if lo <= eps < hi:
                group_times[name].append(t)
                break

    group_avg_time = {}
    for name, ts in group_times.items():
        if ts:
            group_avg_time[name] = sum(ts) / len(ts)
        else:
            group_avg_time[name] = 0.0  # 或者 None，看你喜好

    print("group avg time:", group_avg_time)
    
    
    log_filename = f"{query_file}.log".replace(".bin","")
    with open(LOG_DIRECTORY+log_filename,'a') as f:
        f.write("M,epsilon,cost,ratio\n")
        for i in range(len(cost_list)):
            f.write(f"{M>>20},{eps_list[i]},{cost_list[i]},{h_list[i]}\n")
        
    return eps_list, cost_list
    

def main():
    M = 256*1024*1024
    # data_file = f"books_10M_uint64_unique"
    # query_file = f"books_10M_uint64_unique.query.bin"
    # eps_list,cost_list = getExpectedCostPerEpsilon(ipp=512,seg_size=16,M=M,n=int(1e7),ps=4096,type="sample",
    #                                                data_file=data_file,query_file=query_file,s="all_in_once",
    #                                                A=1.0,B=0)
    
    # data_file = f"fb_10M_uint64_unique"
    # query_file = f"range_query_fb_uu.bin"
    # getExpectedRangeCostPerEpsilon(n=int(1e7),seg_size=16,M=M,ipp=512,ps=4096,query_file=query_file,data_file=data_file,
    #                                A=1.0,B=0.00284)
    
    data_file = f"books_200M_uint64_unique"
    query_file = f"books_200M_uint64_unique.4Mtable2.bin"
    getExpectedJoinCostPerEpsilon(ipp=512,seg_size=16,M=M,n=int(2e7),ps=4096,data_file=data_file,query_file=query_file,
                                   A=1.0,B=0)
    
if __name__ == "__main__":
    main()