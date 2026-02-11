echo "Running with different epsilon and memory values..."
for memory in 10 20 40 60; do
    ./falcon_bench epsilon --dataset books_10M_uint64_unique \
                --keys 10000000 \
                --epsilon 2 4 6 8 10 12 14 16 18 20 24 32 48 64 128\
                --repeats 3 \
                --memory $memory \
                --policy LRU
done
