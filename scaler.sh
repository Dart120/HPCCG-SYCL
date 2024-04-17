#!/bin/bash

# Define base output file path
BASE_OUTPUT_FILE="hpccg_output"

# Compile the program
# Uncomment the next line if you want to clean and recompile every time
# make -f MakefileSYCL clean
make -f MakefileSYCL

# Loop through problem sizes from 50 to 125 in steps of 25
for size in {50..125..25}; do
    # Define the output file for this size
    OUTPUT_FILE="${BASE_OUTPUT_FILE}_${size}.txt"

    # Check if the file exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "File $OUTPUT_FILE exists, appending to it."
    else
        echo "File $OUTPUT_FILE does not exist, will be created."
    fi

    # Run the command 5 times and append the output to the file
    for i in {1..5}; do
        echo "Run #$i for size $size x $size x $size" >> $OUTPUT_FILE
        echo "Run #$i for size $size x $size x $size"
        ./test_HPCCG $size $size $size >> $OUTPUT_FILE
    done
done

