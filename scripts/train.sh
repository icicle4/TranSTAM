TAG=$1
MODEL=$2
LAYERNUM=$3
TRAINHDF5=$4
DATASETDIR=$5

mkdir ./output_${TAG}

DATASETDIR=${DATASETDIR}
TRAINHDF5=${TRAINHDF5}

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --clip_max_norm 0.5 --lr 0.001 --hidden_dim 256 \
--dim_feedforward 1024 --weight_decay 0.0001 --root_dir ${TRAINHDF5} --batch_size 4 --epochs 10 --lr_drop 20 \
--track_history_len 50  --model ${MODEL} --dropout 0.0 --app_dim 256 --tag ${TAG} --output_dir ./output_${TAG} \
--num_workers 0 --test_per_epoch 5


mkdir ./inference_result_${TAG}
mkdir ./inference_result_${TAG}_post/
mkdir ./inference_result_${TAG}_logs

echo "CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR} \
  --output_dir ./inference_result_${TAG} --resume output_${TAG}/checkpoint0009.pth --match_threshold 0.2 --model ${MODEL}
"

CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR} --drop_simple_case \
  --output_dir ./inference_result_${TAG} --resume output_${TAG}/checkpoint0009.pth --match_threshold 0.2 --model ${MODEL}
