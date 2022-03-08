#!/bin/bash

python test.py --enc_layer_num 2 --dec_layer_num 2 --app_dim 256 --pos_dim 4 \
  --track_history_len 150 --dim_feedforward 1024 --hidden_dim 256 --dropout 0.0 --nheads 8 --drop_simple_case \
  --to_inference_pb_dir ./data/TMOH_17_pbs_pca --with_abs_pe "with_abs_pe" --with_relative_pe "with_relative_pe" --with_assignment_pe "with_assignment_pe" \
  --output_dir ./inference_result --resume ./data/model_weights/checkpoint0009.pth --match_threshold 0.2 --model "trans_stam"

python scripts/post_process_trajectory.py --predict_dir ./inference_result \
--out_path ./inference_result_post/ --not_include_move_camera 1

echo "Tracking Result is saved in ./inference_result_post/"