import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq  # 用于解非线性方程
from scipy.special import zeta     # Riemann zeta 函数
from collections import Counter

alpha = 1
DATASETS_DIRECTORY = "/mnt/home/zwshi/Datasets/SOSD/"

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
      query_file: 二进制文件路径或 numpy array of query keys (uint64)
      data: 已排序的 data keys (np.array dtype uint64) OR just integer N (total positions)
      epsilon: int
      ipp: items per page
    returns:
      page_counts: np.array length num_pages, expected counts per page (sum ~= Q * (2eps+1))
      T_pos: np.array length N, expected counts per position
    """
    # 读取 queries 支持文件或直接数组
    
    if isinstance(query_file, str):
        queries = np.fromfile(query_file, dtype=np.uint64)
    else:
        queries = np.asarray(query_file, dtype=np.uint64)

    Q = len(queries)

    # 如果 data 是 array，则 N = len(data)，同时把 queries 映成真实位置 pos
    if isinstance(data, np.ndarray):
        N = len(data)
        # pos = index of largest data <= query -> searchsorted -1
        pos = np.searchsorted(data, queries, side='right') - 1
        pos = np.clip(pos, 0, N-1).astype(np.int64)
    else:
        # data 给的是总长度 N（位置空间）
        N = int(data)
        # 如果只有 keys 的分布不可映射到 pos，这里我们假设 queries 已经是位置（0..N-1）
        pos = queries.astype(np.int64)
        pos = np.clip(pos, 0, N-1)

    # 1) 构造真实位置直方图 H (长度 N)
    H = np.bincount(pos, minlength=N).astype(np.float64)  # sum(H) == Q

    # 2) 构造合成核 k = g * h. 我们假设 g = uniform box, h = uniform box => k = triangular
    k = triangular_kernel_from_box(epsilon)  # length K = 4*eps + 1, sums to 1

    # 3) 卷积 H * k -> T positions (期望位置访问次数). 使用 'same' 保持长度 N
    if use_fft:
        # 对于非常大 N 可考虑 FFT 卷积实现，如 scipy.signal.fftconvolve
        from scipy.signal import fftconvolve
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

    return page_counts, T

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

def che_characteristic_time(qs, C):
    # qs: array of popularity q(i)
    print("[*] starting solve characteristic time")
    def f(t):
        return np.sum(1 - np.exp(-qs * t)) - C
    # root-finding to solve C = Σ(1 - e^{-q_i t})
    return brentq(f, 1e-6, 1e6)

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

def expected_DAC(epsilon, ipp):
    return 1 + (2*epsilon/ipp)

def expected_IAC(epsilon, ipp):
    return 1 + (2*epsilon/ipp)

def uniform_ratio(C,N):
    return C / N

def zipf_ratio(C,N,alpha):
    qs = zipf_popularity(N, alpha)
    t_C = che_characteristic_time(qs, C)
    hit_rates = che_hit_rates(qs, t_C)
    return np.sum(qs * hit_rates)

def sample_ratio(C,N,qs):
    t_C = che_characteristic_time(qs, C)
    print("[+] successfully solved characteristic_time")
    hit_rates = che_hit_rates(qs, t_C)
    return np.sum(qs * hit_rates)

def validate_ratio(ratio):
    if ratio >= 1.0:
        h = 1.0
    elif ratio <= 0:
        h = 0.0
    else:
        h = ratio
    return h

def cost_function(epsilon, n, seg_size, M, ipp, ps, type="uniform", query_file="", data_file=""):
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
            page_counts, Tpos = estimate_page_counts_from_queryfile(query_file, data, epsilon, ipp)
            total_page_requests = page_counts.sum()
            q = page_counts / total_page_requests
            q = np.sort(q)[::-1]
            buffer_ratio = sample_ratio(C, total_pages, q)
        elif type == "zipf":
            buffer_ratio = zipf_ratio(C, total_pages,alpha)
        
        h = validate_ratio(buffer_ratio)

    return (1 - h) * expected_DAC(epsilon, ipp), h



def getExpectedCostPerEpsilon(ipp, seg_size, M, n, ps,type="uniform",data_file="",query_file=""):
    eps_list = []
    cost_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    for eps in range(least_eps, 65):
        cost,h = cost_function(eps, n, seg_size, M, ipp, ps, type, query_file, data_file)
        eps_list.append(eps)
        cost_list.append(cost)
    print(eps_list)
    print(cost_list)
    return eps_list, cost_list

def getExpectedCacheHitRatioPerEpsilon(ipp, seg_size, M, n, ps,type="uniform",data_file="",query_file=""):
    eps_list = []
    h_list = []
    least_eps = math.ceil(n*seg_size/(2*M))
    for eps in range(least_eps, 65):
        cost,h = cost_function(eps, n, seg_size, M, ipp, ps, type, query_file, data_file)

        eps_list.append(eps)
        h_list.append(h)
    
    print(eps_list)
    print(h_list)
    return eps_list, h_list


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
    M = 30*1024*1024
    data_file = f"{DATASETS_DIRECTORY}books_20M_uint64_unique"
    query_file = f"{DATASETS_DIRECTORY}books_20M_uint64_unique.query.bin"
    # getOptimalEpsilon(ipp=512,seg_size=16,M=M,n=int(2e7),ps=4096,type="sample")
    eps_list,cost_list = getExpectedCostPerEpsilon(ipp=512,seg_size=16,M=M,n=int(2e7),ps=4096,type="sample",
                                                   data_file=data_file,query_file=query_file)
    
    
if __name__ == "__main__":
    main()