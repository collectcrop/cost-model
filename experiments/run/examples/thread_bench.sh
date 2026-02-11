echo "Running with threads..."
./falcon_bench threads --dataset books_200M_uint64_unique \
                --keys 200000000 \
                --threads 1 2 4 8 16 32 64 128 \
                --repeats 3 \
                --memory 0 \
                --policy NONE
