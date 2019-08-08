#!/bin/bash
# Delete all cache files
find ./ -name "*pycache*" -exec rm -rf {} \;
conda install zip

zip -0 -r $1.zip .