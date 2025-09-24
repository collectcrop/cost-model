
#!/usr/bin/env bash
# pdam_fio_test.sh — measure throughput vs. number of threads (p) using fio
# Usage:
#   chmod +x pdam_fio_test.sh
#   ./pdam_fio_test.sh /path/to/testfile
#   ./pdam_fio_test.sh /dev/nvme0n1 --device
#
# set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <target-file-or-device> [--device]"
    exit 1
fi

TARGET="$1"
IS_DEVICE=0
if [[ "${2:-}" == "--device" ]]; then
    IS_DEVICE=1
fi

JOBS=($(seq 1 8) $(seq 9 2 64))
SIZE_PER_JOB="2G"   # smaller for quicker runs
BS="4k"

IOENGINE="psync"         # io_uring,libaio
if ! fio -v >/dev/null 2>&1; then
    echo "fio is not installed. Please install fio and retry."
    exit 1
fi
# if ! fio --enghelp | grep -q io_uring; then
#     IOENGINE="libaio"
# fi

OUTCSV="pdam_results.csv"
echo "${IOENGINE}"
echo "p,bw_MiBps" > "${OUTCSV}"

for i in {1..8}; do
    for p in "${JOBS[@]}"; do
        echo "=== Running p=${p} ==="
        if [[ ${IS_DEVICE} -eq 1 ]]; then
        FIOCMD=(fio --name=pdam --filename="${TARGET}" --direct=1 --rw=randread --bs="${BS}" \
                    --ioengine="${IOENGINE}" --iodepth=64 --numjobs="${p}" --thread=1 \
                    --group_reporting=1 --randrepeat=0 --norandommap --refill_buffers=1 \
                    --size="${SIZE_PER_JOB}")
        else
        FIOCMD=(fio --name=pdam --filename="${TARGET}" --direct=1 --rw=randread --bs="${BS}" \
                    --ioengine="${IOENGINE}" --iodepth=64 --numjobs="${p}" --thread=1 \
                    --group_reporting=1 --randrepeat=0 --norandommap --refill_buffers=1 \
                    --size="${SIZE_PER_JOB}")
        fi

        OUTPUT="$("${FIOCMD[@]}")"
        echo "${OUTPUT}"
        echo "------------------------------------------------"
        echo "${OUTPUT}" | awk '/READ: bw=/ {print $0}' | sed -n 's/.*bw=\([0-9]\+\(\.[0-9]\+\)\?\)MiB\/s.*/\1/p'
        echo "------------------------------------------------"
        BW=$(echo "${OUTPUT}" | awk '/READ: bw=/ {print $0}' | sed -n 's/.*bw=\([0-9]\+\(\.[0-9]\+\)\?\)MiB\/s.*/\1/p')
        echo "${p},${BW}" | tee -a "${OUTCSV}"
    done
done


echo "Done. Results saved to ${OUTCSV}"