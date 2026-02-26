echo "Running with batches..."
./falcon_bench batch --dataset books_200M_uint64_unique \
                --keys 200000000 \
                --batch 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 \
                --memory 0 \
                --policy NONE
