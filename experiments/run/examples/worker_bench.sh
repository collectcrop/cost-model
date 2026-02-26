echo "Running with batches..."
./falcon_bench worker --dataset books_200M_uint64_unique \
                --keys 200000000 \
                --producers 1 2 4 8 16 32 64 128 \
                --workers 1 2 4 8 16 32 64 128 \
                --memory 0 \
                --policy NONE
