from optimalEpsilon import join_cost_function

M = 10 * 1024 * 1024         
PAGE_SIZE = 4096
KEY_SIZE = 8
ipp = PAGE_SIZE // KEY_SIZE

data_file   = "books_200M_uint64_unique"
join_file   = "books_200M_uint64_unique.1Mtable.bin"
par_file    = "books_200M_uint64_unique.1Mtable.par"
bitmap_file = "books_200M_uint64_unique.1Mtable.bitmap"

eps = 16
for eps in [8, 12, 16, 20, 24, 32, 48, 64, 128]:
    avg_cost, detail = join_cost_function(
        epsilon=eps,
        n=int(2e8),
        seg_size=16,           
        M=M,
        ipp=ipp,
        ps=PAGE_SIZE,
        data_file=data_file,
        join_file=join_file,
        par_file=par_file,
        bitmap_file=bitmap_file,
    )

    print("epsilon =", eps)
    print("avg miss pages per join tuple =", avg_cost)
    print("detail:", detail)
