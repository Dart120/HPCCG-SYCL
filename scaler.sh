#!/bin/bash

# Define the output file path
OUTPUT_FILE="hpccg_output.txt"

# Check if the file exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "File exists, appending to it."
else
    echo "File does not exist, will be created."
fi
make -f MakefileSYCL clean
make -f MakefileSYCL
# Run the command 5 times and append the output to the file
for i in {1..5}; do
    echo "Run #$i" >> $OUTPUT_FILE
    ./test_HPCCG 30 30 30 >> $OUTPUT_FILE
done
