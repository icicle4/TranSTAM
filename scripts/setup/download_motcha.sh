#!/usr/bin/env bash

wget -P data https://motchallenge.net/data/MOT17Det.zip
wget -P data https://motchallenge.net/data/MOT17Labels.zip

unzip -d data/MOT17Labels $DATA_PATH/MOT17Labels.zip
unzip -d data/MOT17Det $DATA_PATH/MOT17Det.zip

