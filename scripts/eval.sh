
EPOCHNUM=$1

EvalEpoch=$(printf "%02d" $[$EPOCH-1])

LABELPATH=/ssd/yqfeng/research/datasets/MOT17/test_gts
DATASETDIR=/ssd/yqfeng/work/datasets/MOT17/public_detections_dim_256/test

mkdir ./inference_result/${EPOCHNUM}
mkdir ./inference_result/${EPOCHNUM}_post
mkdir ./inference_result/logs
mkdir ./inference_result/logs/${EPOCHNUM}


CUDA_VISIBLE_DEVICES=1 python test.py --enc_layer_num 2 --dec_layer_num 2 --app_dim 256 --pos_dim 4 \
  --track_history_len 50 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.1 --nheads 8 \
  --to_inference_pb_dir ${DATASETDIR} \
  --output_dir "./inference_result/${EPOCHNUM}/" --resume output/checkpoint00${EvalEpoch}.pth --match_threshold 0.2 --model "sota"

python3 scripts/eval_motchallenge_test.py \
--groundtruths ${LABELPATH} \
--tests ./inference_result/${EPOCHNUM} \
--eval_official \
--score_threshold -1 | tee ./inference_result/logs/${EPOCHNUM}/MOT17_test.log

python scripts/post_process_trajectory.py --predict_dir ./inference_result/${EPOCHNUM} \
--out_path ./inference_result/${EPOCHNUM}_post --not_include_move_camera 1

python3 scripts/eval_motchallenge_test.py \
--groundtruths ${LABELPATH} \
--tests ./inference_result/${EPOCHNUM} \
--eval_official \
--score_threshold -1 | tee ${EVALUATIONRESDIR}/${TAG}_${DISTHRESHOLD}_${IOUTHRESHOLD}_${COMPETEIOUTHRESHOLD}/MOT17_test_post.log