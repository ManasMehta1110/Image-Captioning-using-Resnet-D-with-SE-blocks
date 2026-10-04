# Decisions log

- 2026-09-30 22:48 IST: SCST stage (loss ablation + headline model) will use the SE-ResNet-50 encoder, chosen because it is closest to the original SE-ResNet-D design. Decided BEFORE any SE-ResNet-50 CE result existed (at this time only ResNet-50 CE had results; SE-ResNet-50 CE had not started).
- 2026-10-02 11:44 IST: scst_mixed_nonorm_se_s0 was early-stopped at epoch 3 (patience 3) while still recovering, whereas every other SCST run trained the full 10 epochs. To give it the same 10-epoch budget, it is resumed from epoch 3 with --patience 10. The paper reports this.
