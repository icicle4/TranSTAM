#!/bin/bash

MODELNAME=$1
ABLATIONTAG=$2
MATCHTHRESHOLD=$3
TRACKHISLEN=$4
LAYERNUM=$5
CACHEWINDOW=$6
SAMPLEWINDOW=$7
CUDANUM=$8
EPOCH=$9
ROOTDIR=${10}

#EvalEpoch=$(printf "%02d" $[$EPOCH-1])

TRACKRESDIR=/ssd/yqfeng/research/datasets/MOT_17_20_mix/test_res
DATASETDIR=${ROOTDIR}/pca_detection/train/
MOT17DIR=/ssd/yqfeng/research/datasets/MOT17
EVALUATIONRESDIR=/root/work/edgetransformerquik/evaluation_reports

mkdir ${TRACKRESDIR}_${ABLATIONTAG}
mkdir ${EVALUATIONRESDIR}/${ABLATIONTAG}

mkdir ./output_${ABLATIONTAG}_Split12

echo "python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split12.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.1 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split12" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW}
"

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split12.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.1 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split12" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW}

CUDA_VISIBLE_DEVICES=${CUDANUM} python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len ${TRACKHISLEN} --dim_feedforward 1024 --hidden_dim 256 --dropout 0.1 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR}/Split3 \
  --output_dir ${TRACKRESDIR}_${ABLATIONTAG} --resume "./output_${ABLATIONTAG}_Split12/checkpoint0004.pth" --match_threshold ${MATCHTHRESHOLD} --model ${MODELNAME}

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split23.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.1 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split23" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW}

CUDA_VISIBLE_DEVICES=${CUDANUM} python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len ${TRACKHISLEN} --dim_feedforward 1024 --hidden_dim 256 --dropout 0.1 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR}/Split1 \
  --output_dir ${TRACKRESDIR}_${ABLATIONTAG} --resume "./output_${ABLATIONTAG}_Split23/checkpoint0004.pth" --match_threshold ${MATCHTHRESHOLD} --model ${MODELNAME}

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split13.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.1 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split13" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW}

CUDA_VISIBLE_DEVICES=${CUDANUM} python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len ${TRACKHISLEN} --dim_feedforward 1024 --hidden_dim 256 --dropout 0.1 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR}/Split2 \
  --output_dir ${TRACKRESDIR}_${ABLATIONTAG} --resume "./output_${ABLATIONTAG}_Split13/checkpoint0004.pth" --match_threshold ${MATCHTHRESHOLD} --model ${MODELNAME}

echo "------------- Evaluation Results ----------"

python3 scripts/eval_motchallenge.py \
--groundtruths ${MOT17DIR}/train \
--tests ${TRACKRESDIR}_${ABLATIONTAG} \
--eval_official \
--score_threshold -1 | tee ${EVALUATIONRESDIR}/${ABLATIONTAG}/MOT17_train.log

echo "------------- Post Process Results ----------"

mkdir ${TRACKRESDIR}_${ABLATIONTAG}_post

python scripts/post_process_trajectory.py \
--predict_dir ${TRACKRESDIR}_${ABLATIONTAG} \
--out_path ${TRACKRESDIR}_${ABLATIONTAG}_post \
--not_include_move_camera 1

echo "------------- Evaluation Results ----------"

python3 scripts/eval_motchallenge.py \
--groundtruths ${MOT17DIR}/train \
--tests ${TRACKRESDIR}_${ABLATIONTAG}_post \
--eval_official \
--score_threshold -1 | tee ${EVALUATIONRESDIR}/${ABLATIONTAG}/MOT17_train_post.log