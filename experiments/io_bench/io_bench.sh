#!/bin/bash
TESTFILE="testfile"
BS=4096       # block size
NREQ=100000   # number of requests
REPEAT=30      # repeat times
DEPTHS=(1 2 4 8 16 32 64 128 256 512 1024) 

# check test file
if [ ! -f "$TESTFILE" ]; then
    echo "generating $TESTFILE ..."
    fallocate -l 4G "$TESTFILE"
fi

# calculate average
calc_avg() {
    awk '{ total += $1; count++ } END { if (count > 0) print total/count; }'
}

run_test() {
    mode=$1
    depth=$2
    qd=$3
    result_file=$4
    echo "[$mode] depth=$depth running..."

    for ((i=1; i<=REPEAT; i++)); do
        if [ "$mode" = "psync" ]; then
            output=$(./io_bench psync $depth 1 testfile)
        elif [ "$mode" = "libaio" ]; then
            output=$(./io_bench libaio $depth $qd testfile)
        else [ "$mode" = "io_uring" ];
            output=$(./io_bench io_uring $depth $qd testfile)
        fi

        echo "$output" | grep "throughput"
        thr=$(echo "$output" | grep "throughput" | awk '{print $5}')
        echo $thr >> "$result_file"
    done
}

echo "================= I/O Benchmark ================="
echo "file: $TESTFILE, block=$BS, requests=$NREQ, repeats=$REPEAT"
echo "================================================="

mkdir -p results

for depth in "${DEPTHS[@]}"; do
    for mode in psync libaio io_uring; do    # psync libaio io_uring
        result_file="results/${mode}_${depth}.txt"
        rm -f "$result_file"

        run_test $mode $depth 128 $result_file
    done
done
