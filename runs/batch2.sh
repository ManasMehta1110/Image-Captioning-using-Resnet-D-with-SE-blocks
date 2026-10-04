cd /c/Projects/Image-Captioning-using-Resnet-D-with-SE-blocks
P=/c/Projects/gpuenv/Scripts/python.exe
( $P evaluate_features.py runs/scst_mixed_noreset_se_s0 > runs/eval_noreset_b7.log 2>&1 && echo "noreset b7 ok $(date +%H:%M)"
  $P evaluate_features.py runs/scst_mixed_noreset_se_s0 --beam 3 --alpha 0 --min_len 0 --block_ngram 0 > runs/eval_noreset_b3.log 2>&1 && echo "noreset b3 ok $(date +%H:%M)"
  for r in ce_seresnet50_s0 scst_mixed_se_s0 scst_pure_se_s0 scst_mixed_noreset_se_s0 ce_resnet50_s0 ce_resnet50d_s0; do
    $P evaluate_features.py runs/$r --beam 1 --alpha 0 --min_len 0 --block_ngram 0 > runs/eval_${r}_greedy.log 2>&1 && echo "$r greedy ok $(date +%H:%M)"; done ) &
$P train_decoder.py --feat_dir data/coco/timm_features/seresnet50 --init runs/ce_seresnet50_s0/best.pt --seed 0 --loss mixed --no_norm_adv --out runs/scst_mixed_nonorm_se_s0 --resume --patience 10 >> runs/scst_mixed_nonorm_se_s0.out 2>&1 && echo "nonorm training done $(date +%H:%M)"
wait
$P evaluate_features.py runs/scst_mixed_nonorm_se_s0 > runs/eval_nonorm_b7.log 2>&1 && echo "nonorm b7 ok $(date +%H:%M)"
$P evaluate_features.py runs/scst_mixed_nonorm_se_s0 --beam 3 --alpha 0 --min_len 0 --block_ngram 0 > runs/eval_nonorm_b3.log 2>&1 && echo "nonorm b3 ok $(date +%H:%M)"
$P evaluate_features.py runs/scst_mixed_nonorm_se_s0 --beam 1 --alpha 0 --min_len 0 --block_ngram 0 > runs/eval_nonorm_greedy.log 2>&1 && echo "nonorm greedy ok $(date +%H:%M)"
echo ALL_DONE
