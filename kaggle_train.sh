python train_resattunet.py \
  --dataset kagglemix \
  --train-anno split_files/file_pairs_train.txt \
  --val-anno split_files/file_pairs_val.txt \
  --log-dir resatt_logs \
  --batch-size 16 \
  --num-workers 4 \
  --epochs 100 \
  --lr 0.001
