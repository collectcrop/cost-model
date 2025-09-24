#!/bin/bash
# 自动化 I/O 基准测试
# 测试 pread (多线程)、libaio (QD)、io_uring (QD)
# 会运行多次并取平均值

TESTFILE="testfile"
BS=4096       # block size
NREQ=100000   # 总请求数
REPEAT=30      # 每组重复次数
DEPTHS=(1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384)  # 并发度 1 2 4 8 16 32 64 128 256 512 1024 2048 4096

# 检查测试文件是否存在
if [ ! -f "$TESTFILE" ]; then
    echo "生成测试文件 $TESTFILE ..."
    fallocate -l 4G "$TESTFILE"
fi

# 用 awk 计算平均值
calc_avg() {
    awk '{ total += $1; count++ } END { if (count > 0) print total/count; }'
}

run_test() {
    mode=$1
    depth=$2
    qd=$3
    result_file=$4
    echo "[$mode] depth=$depth 测试中..."

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
echo "文件: $TESTFILE, block=$BS, 请求数=$NREQ, 重复次数=$REPEAT"
echo "================================================="

mkdir -p results

for depth in "${DEPTHS[@]}"; do
    for mode in psync libaio io_uring; do    #
        result_file="results/${mode}_${depth}.txt"
        rm -f "$result_file"

        run_test $mode $depth 128 $result_file
    done
done
