# Silver transfer learning (gold KTB test set, mean +/- std over 3 seeds)

| Configuration | Lemma F1 | UPOS F1 | Grammeme F1 |
|---|---|---|---|
| Gold-only (from scratch) | 45.68 +/- 0.75 | 69.39 +/- 2.51 | 44.27 +/- 2.76 |
| Silver-pretrain -> gold-finetune (filtered) | 49.33 +/- 0.75 | 70.89 +/- 0.84 | 47.72 +/- 2.40 |
| Silver-pretrain -> gold-finetune (unfiltered) | 48.48 +/- 1.17 | 69.54 +/- 0.51 | 42.45 +/- 1.23 |

Silver pretraining gives a statistically significant lemma-F1 gain (paired t-test p=0.025,
Cohen d=3.6) and a large-effect grammeme trend (d=1.6). Filtering helps on all 3 tasks
(lemma +0.9, UPOS +1.4, grammeme +5.3 pp) vs unfiltered silver, with large effect sizes.