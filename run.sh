#!/bin/bash

# Checking number of arguments
if [ "$#" -lt 2 ]; then
    echo "Invalid Arguments"
    exit 1
fi

if [ "$1" -eq "1" ]; then
    python3 Q1/model.py $2 $3 $4
    exit 0
elif [ "$1" -eq "2" ]; then
    python3 Q2/main.py $2 $3
    exit 0
fi

echo "Invalid Arguments"
exit 1