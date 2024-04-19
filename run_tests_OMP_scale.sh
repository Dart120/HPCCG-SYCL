#!/bin/bash

# Base output file name
BASE_OUTPUT_FILE="runtime_test_OMP"

# Number of threads to use for OpenMP
thread_counts=("1" "2" "4" "8" "16" "32" "64")

# Iterate through the array of thread counts
for cores in "${thread_counts[@]}"; do
    export OMP_NUM_THREADS="$cores"
    echo "Using $OMP_NUM_THREADS threads."

    # Loop from 50 to 175 in steps of 25 for problem sizes
    for i in {50..150..25}; do
        OUTPUT_FILE="${BASE_OUTPUT_FILE}_${i}.txt"

        # Check if the file exists
        if [ -f "$OUTPUT_FILE" ]; then
            echo "File $OUTPUT_FILE exists, appending to it."
        else
            echo "File $OUTPUT_FILE does not exist, will be created."
        fi

        # Run the test multiple times
        for j in {1..5}; do
            echo "Running test_HPCCG with dimensions $i $i $i" >> "$OUTPUT_FILE"
            echo "Running test_HPCCG with dimensions $i $i $i"
            # Use the command 'time' and append both stderr and stdout to the output file
            { time ./test_HPCCG $i $i $i; } 2>> "$OUTPUT_FILE" >> "$OUTPUT_FILE"
        done
    done
done

echo "Done. Results stored for all configurations."
