#!/bin/bash

MODELNAME=$1
ABLATIONTAG=$2
LAYERNUM=$3
CACHEWINDOW=$4
SAMPLEWINDOW=$5
ABSPE=$6
RELATIVEPE=$7
ASSIGNPE=$8
ASPESTYLE=${9}
TRACKRESDIR=${10}
EVALUATIONRESDIR=${11}

ROOTDIR=/root/transtam/data/TMOH_17_pbs_pca_matched_th0.6
MOT17DIR=/root/transtam/data/MOT17/

CUDANUM=0
TRACKHISLEN=150
EPOCH=10

DATASETDIR=${ROOTDIR}/pca_detection/train/

mkdir ${TRACKRESDIR}_${ABLATIONTAG}
mkdir ${EVALUATIONRESDIR}/${ABLATIONTAG}

mkdir ./output_${ABLATIONTAG}_Split12


python3 -m torch.distributed.launch --master_port 29503 --nproc_per_node=8 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split12.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.0 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split12" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW} \
--with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} --aspe_style ${ASPESTYLE}

CUDA_VISIBLE_DEVICES=${CUDANUM} python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len ${TRACKHISLEN} --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 --drop_simple_case \
  --to_inference_pb_dir ${DATASETDIR}/Split3 --with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} \
  --output_dir ${TRACKRESDIR}_${ABLATIONTAG} --resume "./output_${ABLATIONTAG}_Split12/checkpoint0009.pth" --match_threshold 0.2 --model ${MODELNAME} \
  --aspe_style ${ASPESTYLE}

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29503 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split23.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.0 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split23" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW} \
--with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} --aspe_style ${ASPESTYLE}

CUDA_VISIBLE_DEVICES=${CUDANUM} python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len ${TRACKHISLEN} --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 --drop_simple_case \
  --to_inference_pb_dir ${DATASETDIR}/Split1 --with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} \
  --output_dir ${TRACKRESDIR}_${ABLATIONTAG} --resume "./output_${ABLATIONTAG}_Split23/checkpoint0009.pth" \
  --match_threshold 0.2 --model ${MODELNAME} --aspe_style ${ASPESTYLE}

python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 29503 --use_env train.py --enc_layer_num ${LAYERNUM} \
--dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir "${ROOTDIR}/train_Split13.hdf5" --batch_size 4 --epochs ${EPOCH} --lr_drop 20 \
--track_history_len ${SAMPLEWINDOW}  --model ${MODELNAME} --dropout 0.0 --app_dim 256 --tag "ablation_study" \
--output_dir "./output_${ABLATIONTAG}_Split13" --num_workers 0 --test_per_epoch 5 --cache_window_size ${CACHEWINDOW} \
--with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} --aspe_style ${ASPESTYLE}

CUDA_VISIBLE_DEVICES=${CUDANUM} python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len ${TRACKHISLEN} --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 --drop_simple_case \
  --to_inference_pb_dir ${DATASETDIR}/Split2 --with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} \
  --output_dir ${TRACKRESDIR}_${ABLATIONTAG} --resume "./output_${ABLATIONTAG}_Split13/checkpoint0009.pth" \
  --match_threshold 0.2 --model ${MODELNAME} --aspe_style ${ASPESTYLE}

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