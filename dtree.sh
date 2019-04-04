#!/bin/bash

# Checking number of arguments
if [ "$#" -lt 4 ]; then
    echo "Invalid Arguments"
    exit 1
fi

python3 Q1/main.py $1 $2 $3 $4