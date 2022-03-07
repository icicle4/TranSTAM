TAG=$1
MODEL=$2
LAYERNUM=$3
THRESHOLD=$4

mkdir ./output_${TAG}

LABELPATH=/ssd/yqfeng/research/datasets/MOT17/test_gts
DATASETDIR=/ssd/yqfeng/research/datasets/results_reid_wo_traindata/pca_detection/test


mkdir ./inference_result_${TAG}_${THRESHOLD}
mkdir ./inference_result_${TAG}_${THRESHOLD}_post/
mkdir ./inference_result_${TAG}_${THRESHOLD}_logs

echo "CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR} \
  --output_dir ./inference_result_${TAG} --resume output_${TAG}/checkpoint0009.pth --match_threshold ${THRESHOLD} --model ${MODEL}
"

CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR} --drop_simple_case \
  --output_dir ./inference_result_${TAG}_${THRESHOLD} --resume output_${TAG}/checkpoint0009.pth --match_threshold ${THRESHOLD} --model ${MODEL}

python3 scripts/eval_motchallenge_test.py \
--groundtruths ${LABELPATH} \
--tests ./inference_result_${TAG}_${THRESHOLD} \
--eval_official \
--score_threshold -1 | tee ./inference_result_${TAG}_${THRESHOLD}_logs/MOT17_test.log

python scripts/post_process_trajectory.py --predict_dir ./inference_result_${TAG}_${THRESHOLD} \
--out_path ./inference_result_${TAG}_${THRESHOLD}_post/ --not_include_move_camera 1

python3 scripts/eval_motchallenge_test.py \
--groundtruths ${LABELPATH} \
--tests ./inference_result_${TAG}_${THRESHOLD}_post/ \
--eval_official \
--score_threshold -1 | tee ./inference_result_${TAG}_${THRESHOLD}_logs/MOT17_test_post.log