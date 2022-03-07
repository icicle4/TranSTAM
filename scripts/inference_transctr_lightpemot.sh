bash scripts/inference_with_wo_trained_reid_light_pemot_transctr.sh "light_pemot_transctr" "light_pemot" 2 "light_pemot" 2 "with_abs_pe" "with_relative_pe" "with_assignment_pe" "diff"

echo "light_pemot_transctr"
cat ./inference_result_light_pemot_transctr_logs/MOT17_test_post.log | tail -n 1