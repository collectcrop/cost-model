echo "single run for point probes test..."
./falcon_bench rangeSig --dataset books_200M_uint64_unique \
            --keys 200000000 \
            --query books_200M_uint64_unique.1Mtable5.bin \
            --memory 0 \
            --policy NONE
