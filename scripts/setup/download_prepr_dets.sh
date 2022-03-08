#!/usr/bin/env bash

cd ./data || exit
curl https://transfer.sh/c8gGQg/prepare_detections.zip -o prepare_detections.zip
unzip prepare_detections.zip