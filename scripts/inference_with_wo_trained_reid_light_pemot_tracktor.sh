TAG=$1
MODEL=$2
LAYERNUM=$3
ABSPE=$4
RELATIVEPE=$5
ASSIGNPE=$6

mkdir ./output_${TAG}

LABELPATH=/ssd/yqfeng/research/datasets/MOT16/test_gts
DATASETDIR=/ssd/yqfeng/Tracktor16_Pbs_Pca/


mkdir ./inference_result_${TAG}
mkdir ./inference_result_${TAG}_post/
mkdir ./inference_result_${TAG}_logs

echo "CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 --drop_simple_case \
  --to_inference_pb_dir ${DATASETDIR} --with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} \
  --output_dir ./inference_result_${TAG} --resume output_${TAG}/checkpoint0009.pth --match_threshold 0.2 --model ${MODEL}"


CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num ${LAYERNUM} --dec_layer_num ${LAYERNUM} --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 --drop_simple_case \
  --to_inference_pb_dir ${DATASETDIR} --with_abs_pe ${ABSPE} --with_relative_pe ${RELATIVEPE} --with_assignment_pe ${ASSIGNPE} \
  --output_dir ./inference_result_${TAG} --resume output_${TAG}/checkpoint0009.pth --match_threshold 0.2 --model ${MODEL}

python3 scripts/eval_motchallenge_test.py \
--groundtruths ${LABELPATH} \
--tests ./inference_result_${TAG} \
--eval_official \
--score_threshold -1 | tee ./inference_result_${TAG}_logs/MOT17_test.log

python scripts/post_process_trajectory.py --predict_dir ./inference_result_${TAG} \
--out_path ./inference_result_${TAG}_post/ --not_include_move_camera 1

python3 scripts/eval_motchallenge_test.py \
--groundtruths ${LABELPATH} \
--tests ./inference_result_${TAG}_post/ \
--eval_official \
--score_threshold -1 | tee ./inference_result_${TAG}_logs/MOT17_test_post.log
