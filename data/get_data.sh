#!/bin/bash

# 1. Create necessary directories
mkdir -p iam/forms
mkdir -p iam/xml
mkdir -p iam/labels

# 2. Download the data using your cookies 
# Note: Ensure 'cookies.txt' is in the current directory or provide the full path!
wget --load-cookies cookies.txt -c --no-check-certificate https://fki.tic.heia-fr.ch/DBs/iamDB/data/formsA-D.tgz
wget --load-cookies cookies.txt -c --no-check-certificate https://fki.tic.heia-fr.ch/DBs/iamDB/data/formsE-H.tgz
wget --load-cookies cookies.txt -c --no-check-certificate https://fki.tic.heia-fr.ch/DBs/iamDB/data/formsI-Z.tgz
wget --load-cookies cookies.txt -c --no-check-certificate https://fki.tic.heia-fr.ch/DBs/iamDB/data/xml.tgz

# 3. Extract the files directly into their respective folders
tar -xzf formsA-D.tgz -C iam/forms
tar -xzf formsE-H.tgz -C iam/forms
tar -xzf formsI-Z.tgz -C iam/forms
tar -xzf xml.tgz -C iam/xml

# 4. DELETE the .tgz files so they don't end up in your final dataset
rm *.tgz