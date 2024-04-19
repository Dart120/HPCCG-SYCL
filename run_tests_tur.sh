#!/bin/bash

# Output file for the time command results
output_file="runtime_test_SYCL_tur.txt"

# Clear the output file if it already exists
> "$output_file"

# Loop from 50 to 300 in steps of 50
for i in {50..300..25}; do
    for j in {0..4..1}; do
        echo "Running test_HPCCG with dimensions $i $i $i" >> "$output_file"
        echo "Running test_HPCCG with dimensions $i $i $i"
        # Use the command 'time' and append both stderr and stdout to the output file
        { time ./test_HPCCG $i $i $i ; } 2>> "$output_file" >> "$output_file"
    done
done

echo "Done. Results stored in $output_file"
