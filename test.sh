#!/usr/bin/env bash
set -euo pipefail

BIN="./test"
DATA="/mnt/backup_disk/backup_2025_full/zwshi/Datasets/SOSD/books_200M_uint64_unique"
QUERIES="/mnt/backup_disk/backup_2025_full/zwshi/Datasets/SOSD/books_200M_uint64_unique.1Kquery.bin"
OUTDIR="log"
mkdir -p "${OUTDIR}"
LOGFILE="${OUTDIR}/qps_raw_$(date +%Y%m%d_%H%M%S).log"
echo "Writing raw outputs to: ${LOGFILE}" | tee -a "${LOGFILE}"
echo | tee -a "${LOGFILE}"

STRATEGIES=("all" "one") #"one"
THREADS=(1 4 16 64 256)    # 1 4 16
REPEAT=10

# Extract qps from stdout. Expected token: qps=<number>
extract_qps() {
  grep -oE 'qps=[0-9]+(\.[0-9]+)?' | tail -n 1 | cut -d'=' -f2
}

# Compute mean and sample stddev from a list of numbers.
# Usage: stats < numbers.txt
stats() {
  awk '
    { x[NR]=$1; s+=$1; s2+=$1*$1 }
    END {
      n=NR;
      if (n==0) { print "mean=NA stddev=NA"; exit 0; }
      mean=s/n;
      if (n==1) { printf("mean=%.6f stddev=0.000000\n", mean); exit 0; }
      var=(s2 - s*s/n)/(n-1);  # sample variance
      if (var < 0) var=0;
      std=sqrt(var);
      printf("mean=%.6f stddev=%.6f\n", mean, std);
    }'
}

echo "BIN=${BIN}"
echo "DATA=${DATA}"
echo "QUERIES=${QUERIES}"
echo "REPEAT=${REPEAT}"
echo

for strat in "${STRATEGIES[@]}"; do
  for th in "${THREADS[@]}"; do
    echo "============================================================"
    echo "Running: strategy=${strat}, threads=${th}"
    echo "============================================================"

    tmpfile="$(mktemp)"
    trap 'rm -f "$tmpfile"' EXIT

    for r in $(seq 1 "$REPEAT"); do
      echo "---- run ${r}/${REPEAT} ----"
      out="$(
        "${BIN}" \
          --data "${DATA}" \
          --queries "${QUERIES}" \
          --strategy "${strat}" \
          --threads "${th}" \
          2>&1 | tee /dev/stderr
      )"

      qps="$(printf "%s\n" "$out" | extract_qps || true)"
      if [[ -z "${qps}" ]]; then
        echo "ERROR: failed to extract qps from output in run ${r} (strategy=${strat}, threads=${th})" >&2
        echo "Make sure your program prints token like: qps=12345.67" >&2
        exit 1
      fi
      echo "${qps}" >> "${tmpfile}"
    done

    echo | tee -a "${LOGFILE}"
    echo "[qps samples]" | tee -a "${LOGFILE}"
    nl -ba "${tmpfile}" | tee -a "${LOGFILE}"

    echo | tee -a "${LOGFILE}"
    echo "[summary] strategy=${strat}, threads=${th}, repeats=${REPEAT}" | tee -a "${LOGFILE}"
    stats < "${tmpfile}" | tee -a "${LOGFILE}"
    echo | tee -a "${LOGFILE}"

    rm -f "${tmpfile}"
    trap - EXIT
  done
done