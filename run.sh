#!/bin/sh

mkdir data
wget https://techassessment.blob.core.windows.net/aiap14-assessment-data/fishing.db -O ./data/fishing.db

python ./src/data_ingestion.py