import math
import matplotlib.pyplot as plt
import numpy as np
import time
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

def join_cost_function(epsilon, n, seg_size, M, ipp, ps,
                       data_file="", join_file="", par_file="", bitmap_file="",
                       s="all_in_once"):
    """Estimate average miss pages per *original join tuple* under hybrid join.

    Parameters
    ----------
    epsilon : int
        PGM epsilon.
    n : int
        Number of keys in the inner table (data_file).
    seg_size : int
        Segment size used by PGM.
    M : int
        Total memory budget in bytes.
    ipp : int
        Items per page (page_size / key_size).
    ps : int
        Page size in bytes.
    data_file : str
        Inner table data filename (relative to DATASETS_DIRECTORY).
    join_file : str
        Generated join-table probe filename (.bin, uint64 keys).
    par_file : str
        Partition length filename (.par, int64, one N per partition).
    bitmap_file : str
        Partition strategy filename (.bitmap, int8, 0=point,1=range).
    s : str
        Probe mode for expected_DAC, "all_in_once" or "one_by_one".

    Returns
    -------
    avg_cost : float
        Average *missed* pages per original join tuple under the hybrid strategy.
    detail : dict
        Extra information (point / range breakdown).
    """
    # Resolve file paths
    data_path   = f"{DATASETS_DIRECTORY}{data_file}"
    join_path   = f"{DATASETS_DIRECTORY}{join_file}"
    par_path    = f"{DATASETS_DIRECTORY}{par_file}"
    bitmap_path = f"{DATASETS_DIRECTORY}{bitmap_file}"

    # Load data & queries
    data = np.fromfile(data_path, dtype=np.uint64)
    queries = np.fromfile(join_path, dtype=np.uint64)

    lengths = np.fromfile(par_path, dtype=np.int64)
    bitmap  = np.fromfile(bitmap_path, dtype=np.int8)

    if len(lengths) != len(bitmap):
        print("[join_cost_function] Warning: len(par) != len(bitmap)")
    Q_total = int(lengths.sum())
    if Q_total != len(queries):
        print("[join_cost_function] Warning: sum(par) != #queries, using #queries as total.")
        Q_total = len(queries)

    N = len(data)
    total_pages = math.ceil(N / ipp)

    # Inner-table page id for each key
    key_to_page = np.arange(N) // ipp

    # Map each join key to page id (like join_partition)
    idx = np.searchsorted(data, queries)
    idx = np.clip(idx, 0, N - 1)
    query_pages = key_to_page[idx]
    pageCount = np.bincount(query_pages, minlength=total_pages)

    # Reconstruct contiguous non-zero page partitions
    partitions = []
    start = None
    for i, cnt in enumerate(pageCount):
        if cnt > 0 and start is None:
            start = i
        elif cnt == 0 and start is not None:
            partitions.append((start, i - 1))
            start = None
    if start is not None:
        partitions.append((start, total_pages - 1))

    if len(partitions) != len(lengths):
        print(f"[join_cost_function] Warning: partition count mismatch, computed={len(partitions)}, file={len(lengths)}")

    # Split workload into:
    #   - point_keys: all queries that fall into bitmap==0 partitions
    #   - ranges: one [lo,hi] per bitmap==1 partition
    point_keys_list = []
    lo_keys = []
    hi_keys = []

    for part_idx, (l, r) in enumerate(partitions):
        mask = (query_pages >= l) & (query_pages <= r)
        N_here = int(mask.sum())
        if part_idx < len(lengths) and N_here != lengths[part_idx]:
            print(f"[join_cost_function] Warning: partition {part_idx} length mismatch, computed={N_here}, par={lengths[part_idx]}")
        is_range = (part_idx < len(bitmap) and bitmap[part_idx] == 1)

        if not is_range:
            # Point partition: keep original join keys
            part_queries = queries[mask]
            if part_queries.size > 0:
                point_keys_list.append(part_queries)
        else:
            # Range partition: approximate as a single range [lo, hi] that covers its pages
            page_lo = l * ipp
            page_hi = min((r + 1) * ipp, N) - 1
            lo_keys.append(data[page_lo])
            hi_keys.append(data[page_hi])

    if len(point_keys_list) > 0:
        point_queries = np.concatenate(point_keys_list)
    else:
        point_queries = np.array([], dtype=np.uint64)

    lo_keys = np.asarray(lo_keys, dtype=np.uint64)
    hi_keys = np.asarray(hi_keys, dtype=np.uint64)

    Q_point = len(point_queries)
    Q_range = len(lo_keys)  # number of range partitions

    # Compute cache capacity (same as in cost_function / range_cost_function)
    M_index  = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C = M_buffer / ps
    if C <= 0:
        # No buffer: miss everything
        cost_point = expected_DAC(epsilon, ipp, s) if Q_point > 0 else 0.0
        if Q_range > 0:
            # Approximate each range partition as scanning all its pages once,
            # without extra overlap benefit.
            # First recompute RDAC for ranges.
            pos_lo = np.searchsorted(data, lo_keys, side='right') - 1
            pos_hi = np.searchsorted(data, hi_keys, side='right') - 1
            RDAC = pos_hi / ipp - pos_lo / ipp + 1 + 2 * epsilon / ipp
            cost_range = float(np.mean(RDAC))
        else:
            cost_range = 0.0
    else:
        # --- Point part: use Che-based model on sampled workload if exists ---
        if Q_point > 0:
            page_counts_p, Tpos_p, Qp_check = estimate_page_counts_from_queryfile(
                point_queries, data, epsilon, ipp
            )
            total_req_p = page_counts_p.sum()
            if total_req_p > 0:
                q_p = page_counts_p / total_req_p
                q_p = np.sort(q_p)[::-1]
                buffer_ratio_p = sample_ratio(C, total_pages, q_p, Qp_check)
                h_p = validate_ratio(buffer_ratio_p)
            else:
                h_p = 0.0
            cost_point = (1.0 - h_p) * expected_DAC(epsilon, ipp, s)
        else:
            cost_point = 0.0

        # --- Range part: reuse range_cost_function's logic on synthetic ranges ---
        if Q_range > 0:
            pos_lo = np.searchsorted(data, lo_keys, side='right') - 1
            pos_hi = np.searchsorted(data, hi_keys, side='right') - 1
            RDAC = pos_hi / ipp - pos_lo / ipp + 1 + 2 * epsilon / ipp

            page_counts_r = estimate_page_counts_from_range_queryfile(
                lo_keys, hi_keys, data, epsilon, ipp
            )
            total_r = sum(page_counts_r.values())
            if total_r > 0:
                q_r = np.array([f / total_r for f in page_counts_r.values()],
                               dtype=np.float64)
                q_r = np.sort(q_r)[::-1]
                buffer_ratio_r = sample_ratio(C, total_pages, q_r)
                h_r = validate_ratio(buffer_ratio_r)
            else:
                h_r = 0.0
            cost_range = (1.0 - h_r) * float(np.mean(RDAC))
        else:
            cost_range = 0.0

    # Hybrid: total miss pages divided by original join tuples
    if Q_total == 0:
        avg_cost = 0.0
    else:
        total_miss_pages = Q_point * cost_point + Q_range * cost_range
        avg_cost = total_miss_pages / Q_total

    detail = {
        "avg_cost": avg_cost,
        "cost_point_per_query": cost_point,
        "cost_range_per_range": cost_range,
        "Q_total": int(Q_total),
        "Q_point": int(Q_point),
        "Q_range": int(Q_range),
    }
    return avg_cost, detail

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
    
    log_filename = f"{query_file}.log".replace(".bin","")
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

def getOptimalEpsilon(ipp, seg_size, M, n, ps,type="uniform"):
    best_cost = float('inf')
    best_eps = None

    least_eps = math.ceil(n*seg_size/(2*M))
    print(f"least_eps: {least_eps}")
    for eps in range(least_eps, 256):
        cost,h = cost_function(eps, n, seg_size, M, ipp, ps, type)
        print(f"eps: {eps}, cost: {cost}")
        if cost < best_cost:
            best_cost = cost
            best_eps = eps
    print(f"best_eps: {best_eps}")
    print(f"best_cost: {best_cost}")
    

def main():
    M = 60*1024*1024
    # data_file = f"books_10M_uint64_unique"
    # query_file = f"books_10M_uint64_unique.query.bin"
    # eps_list,cost_list = getExpectedCostPerEpsilon(ipp=512,seg_size=16,M=M,n=int(1e7),ps=4096,type="sample",
    #                                                data_file=data_file,query_file=query_file,s="all_in_once",
    #                                                A=1.0,B=0)
    
    data_file = f"fb_10M_uint64_unique"
    query_file = f"range_query_fb_uu.bin"
    getExpectedRangeCostPerEpsilon(n=int(1e7),seg_size=16,M=M,ipp=512,ps=4096,query_file=query_file,data_file=data_file,
                                   A=1.0,B=0.00284)
if __name__ == "__main__":
    main()