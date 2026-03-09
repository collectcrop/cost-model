#!/usr/bin/env bash
set -euo pipefail

############################################
# Configuration
############################################

DATASET="books_200M_uint64_unique"
KEYS=200000000
MAX_EXP=10
REPEAT=3

############################################
# Helper
############################################

run_cmd () {
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
    eval "$2"
    echo
}

echo "============================================================"
echo "Running baseline benchmarks"
echo "Dataset : ${DATASET}"
echo "Keys    : ${KEYS}"
echo "MaxExp  : ${MAX_EXP}"
echo "Repeat  : ${REPEAT}"
echo "============================================================"
echo

############################################
# PGM-disk
############################################

run_cmd \
"Point queries test for PGM-disk..." \
"./pgm_disk_test \
    --dataset ${DATASET} \
    --keys ${KEYS} \
    --max-exp ${MAX_EXP} \
    --repeat ${REPEAT}"

run_cmd \
"Range queries test for PGM-disk..." \
"./pgm_disk_range_test \
    --dataset ${DATASET} \
    --keys ${KEYS} \
    --repeat ${REPEAT}"

############################################
# AULID
############################################

run_cmd \
"Point queries test for AULID..." \
"./aulid_test \
    --dataset ${DATASET} \
    --keys ${KEYS} \
    --max-exp ${MAX_EXP} \
    --repeat ${REPEAT}"

run_cmd \
"Range queries test for AULID..." \
"./aulid_range_test \
    --dataset ${DATASET} \
    --keys ${KEYS} \
    --max-exp ${MAX_EXP} \
    --repeat ${REPEAT}"

############################################
# B+Tree
############################################

run_cmd \
"Point queries test for B+Tree..." \
"./bplustree_test \
    --dataset ${DATASET} \
    --keys ${KEYS} \
    --max-exp ${MAX_EXP} \
    --repeat ${REPEAT}"

run_cmd \
"Range queries test for B+Tree..." \
"./bplustree_range_test \
    --dataset ${DATASET} \
    --keys ${KEYS} \
    --max-exp ${MAX_EXP} \
    --repeat ${REPEAT}"

echo "============================================================"
echo "All baseline benchmarks finished."
echo "============================================================"