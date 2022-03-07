#!/usr/bin/env bash

DATA_PATH=$(python -c "from mot_neural_solver.path_cfg import DATA_PATH; print(DATA_PATH)")
wget -P data https://motchallenge.net/data/2DMOT2015.zip
wget -P data https://motchallenge.net/data/MOT17Det.zip
wget -P data https://motchallenge.net/data/MOT17Labels.zip

unzip -d data $DATA_PATH/2DMOT2015.zip
unzip -d data/MOT17Labels $DATA_PATH/MOT17Labels.zip
unzip -d data/MOT17Det $DATA_PATH/MOT17Det.zip
rm data/{MOT17Labels,MOT17Det,2DMOT2015}.zip
