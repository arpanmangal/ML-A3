#!/bin/bash

# Checking number of arguments
if [ "$#" -lt 3 ]; then
    echo "Invalid Arguments"
    exit 1
fi

python3 Q2/main.py 'nn' $1 $2 $3