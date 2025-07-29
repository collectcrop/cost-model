import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq  # 用于解非线性方程
from scipy.special import zeta     # Riemann zeta 函数

alpha = 1
def zipf_popularity(N, alpha):
    norm_const = sum(1 / (i ** alpha) for i in range(1, N + 1))
    return np.array([1 / (i ** alpha) / norm_const for i in range(1, N + 1)])

def che_characteristic_time(qs, C):
    # qs: array of popularity q(i)
    def f(t):
        return np.sum(1 - np.exp(-qs * t)) - C
    # root-finding to solve C = Σ(1 - e^{-q_i t})
    return brentq(f, 1e-6, 1e6)

def che_hit_rates(qs, t_C):
    return 1 - np.exp(-qs * t_C)

# def expected_DAC(epsilon, ipp):
#     dac = 0
#     for k in range(ipp + 1):
#         term = 1 + math.ceil((epsilon - k) / ipp) + math.ceil((epsilon - ipp + k) / ipp)
#         dac += term
#     return dac / ipp

def expected_DAC(epsilon, ipp):
    return 1 + (2*epsilon/ipp)

def uniform_ratio(C,N):
    return C / N

def zipf_ratio(C,N,alpha):
    qs = zipf_popularity(N, alpha)
    t_C = che_characteristic_time(qs, C)
    hit_rates = che_hit_rates(qs, t_C)
    return np.sum(qs * hit_rates)

def cost_function(epsilon, n, seg_size, M, ipp, ps,type="uniform"):
    M_index = n * seg_size / (2 * epsilon)
    M_buffer = M - M_index
    C = M_buffer/ps
    total_pages = math.ceil(n / ipp)
    if (C==0.0):
        h = 0.0
    else:
        if type == "uniform":
            buffer_ratio = uniform_ratio(C, total_pages)
        elif type == "zipf":
            buffer_ratio = zipf_ratio(C, total_pages,alpha)
        
        # print("epsilon: ",epsilon,"ratio",buffer_ratio)
        if buffer_ratio >= 1.0:
            h = 1.0
        elif buffer_ratio <= 0:
            h = 0.0
        else:
            h = buffer_ratio

    return (1 - h) * expected_DAC(epsilon, ipp)

def test_M():
    n = 1000000
    seg_size = 16  # bytes per segment
    ps = 4096       # page size in bytes
    ipp = ps // 16   # 假设每个 item 是 16 bytes
    # M = 10 * 1024 * 1024  # 10MB memory budget
    M_values = [1e5, 5e5, 1e6, 5e6, 1e7, 1.5e7]

    eps_list = []
    cost_list = []

    # best_cost = float('inf')
    # best_eps = None

    for M in M_values:
        least_eps = math.ceil(n*seg_size/(2*M))
        print(f"least_eps: {least_eps}")
        eps_list.clear()
        cost_list.clear()
        for eps in range(least_eps, 256):
            cost = cost_function(eps, n, seg_size, M, ipp, ps)
            eps_list.append(eps)
            cost_list.append(cost)
        plt.plot(eps_list, cost_list, label=f'M={M/1e6}MB')
    plt.title('Cost vs ε under different Memory Budgets')
    plt.xlabel("ε")
    plt.ylabel("Expected I/O Cost")
    shared_params = f"Zipf Distribution\nipp = {ipp}\nn = {n}\nsegment_size = {seg_size}B\n"
    plt.text(0.95, 0.05, shared_params, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right'
            )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def test_n():  
    seg_size = 16  
    ps = 4096       
    ipp = ps // 16   
    M = 1 * 1024 * 1024    # 1M 
    n_list = [1e5,2e5,3e5,4e5,5e5,6e5,7e5,8e5,9e5,1e6]
    eps_list = []
    cost_list = []
    for n in n_list:
        least_eps = math.ceil(n*seg_size/(2*M))
        eps_list.clear()
        cost_list.clear()
        for eps in range(least_eps, 256):
            cost = cost_function(eps, n, seg_size, M, ipp, ps)
            eps_list.append(eps)
            cost_list.append(cost)
        plt.plot(eps_list, cost_list, label=f'n={int(n)}')

    plt.title('Cost vs ε under different n')
    plt.xlabel("ε")
    plt.ylabel("Expected I/O Cost")
    shared_params = f"Zipf Distribution\nipp = {ipp}\nsegment_size = {seg_size}B\nM = {M/1024/1024}MB"
    plt.text(0.95, 0.05, shared_params, transform=plt.gca().transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right'
            )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def getOptimalEpsilon(ipp, seg_size, M, n, ps,type="uniform"):
    best_cost = float('inf')
    best_eps = None

    least_eps = math.ceil(n*seg_size/(2*M))
    print(f"least_eps: {least_eps}")
    for eps in range(least_eps, 256):
        cost = cost_function(eps, n, seg_size, M, ipp, ps, type)
        print(f"eps: {eps}, cost: {cost}")
        if cost < best_cost:
            best_cost = cost
            best_eps = eps
    print(f"best_eps: {best_eps}")
    print(f"best_cost: {best_cost}")
    

def main():
    M = 4*1024*1024
    getOptimalEpsilon(ipp=256,seg_size=16,M=M,n=1000000,ps=4096,type="zipf")
    
if __name__ == "__main__":
    main()