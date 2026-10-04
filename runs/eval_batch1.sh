cd /c/Projects/Image-Captioning-using-Resnet-D-with-SE-blocks
P=/c/Projects/gpuenv/Scripts/python.exe
for r in ce_resnet50_s0 ce_seresnet50_s0 ce_resnet50d_s0 scst_mixed_se_s0 scst_pure_se_s0; do
  $P evaluate_features.py runs/$r > runs/eval_${r}_b7.log 2>&1 && echo "$r beam7 ok $(date +%H:%M)" || echo "$r beam7 FAILED"
  $P evaluate_features.py runs/$r --beam 3 --alpha 0 --min_len 0 --block_ngram 0 > runs/eval_${r}_b3.log 2>&1 && echo "$r beam3 ok $(date +%H:%M)" || echo "$r beam3 FAILED"
done
echo ALL_DONE
