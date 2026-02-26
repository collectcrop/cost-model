echo "Hybrid Join testing..."
./falcon_bench join --dataset books_200M_uint64_unique \
            --keys 200000000 \
            --query books_200M_uint64_unique.1Mtable5 \
            --memory 0 \
            --policy NONE
