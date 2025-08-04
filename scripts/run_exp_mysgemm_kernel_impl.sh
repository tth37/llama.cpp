#!/bin/bash

# Usage help
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sparsity_ratio> <sub_batches>"
    echo "  <sparsity_ratio>: e.g., 0.5"
    echo "  <sub_batches>: e.g., 4"
    exit 1
fi

sparsity_ratio="$1"
sub_batches="$2"

# Configurations: [B, M, N]
configs=(
    "32 1024 1024"
    "32 1024 4096"
    "32 4096 1024"
    "32 2048 2048"
    "32 2048 6144"
    "32 6144 2048"
)

function run_and_average() {
    local cmd="$1"
    local sum=0.0
    for i in {1..3}; do
        output=$($cmd)
        ms=$(echo "$output" | grep "Average latency per run:" | awk '{print $(NF-1)}')
        sum=$(awk "BEGIN {print $sum + $ms}")
    done
    avg=$(awk "BEGIN {print $sum / 3}")
    echo "$avg"
}

# Benchmark loop
for cfg in "${configs[@]}"; do
    read -r B M N <<< "$cfg"
    echo "Benchmarking configuration: B=$B M=$M N=$N"

    # llamafile (baseline)
    llama_cmd="./build/bin/bench-llamafile $B $M $N"
    llama_avg=$(run_and_average "$llama_cmd")
    echo "  [llamafile] avg latency: $llama_avg ms"

    # mysgemm
    mysgemm_cmd="./build/bin/bench-mysgemm $B $M $N $sparsity_ratio $sub_batches"
    mysgemm_avg=$(run_and_average "$mysgemm_cmd")
    echo "  [mysgemm  ] avg latency: $mysgemm_avg ms"
    echo ""
done
