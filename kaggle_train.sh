# kaggle training samples: 
# torchrun --nproc_per_node=8 vit_train_ddp.py -ds kagglemix -t file_pairs_train.txt -v file_pairs_val.txt --tensorboard --batch-size 4 -o kaggle_samples_output
# torchrun --nproc_per_node=8 vit_train_ddp.py -ds kagglemix -t kaggle_train.txt -v kaggle_val.txt --tensorboard --batch-size 16 -o kaggle_samples_output

# torchrun --nproc_per_node=8 vit_train_ddp.py \
#   -ds kagglemix \
#   -t kaggle_train.txt \
#   -v kaggle_val.txt \
#   --tensorboard \
#   --batch-size 8 \
#   --epochs 50 \
#   --workers 2 \
#   -o kaggle_samples_output_v2


  # torchrun --nproc_per_node=8 vit_train_ddp.py \
  # -ds kagglemix \
  # -t file_pairs_train.txt \
  # -v file_pairs_val.txt \
  # --tensorboard \
  # --batch-size 8 \
  # --epochs 50 \
  # --workers 2 \
  # -o kaggle_samples_output_v2

  torchrun --nproc_per_node=8 invnet_train_ddp.py \
  --model InversionNet \
  --device cuda \
  --dataset kagglemix \
  --anno-path split_files \
  --train-anno kaggle_train.txt \
  --val-anno kaggle_val.txt \
  --output-path Invnet_models_full \
  --save-name fcn_l1loss\
  --tensorboard \
  --batch-size 8 \
  --workers 2 \
  --sync-bn \
  --print-freq 20

 
  torchrun --nproc_per_node=8 uvit_train_ddp.py \
  --model UViT \
  --dataset kagglemix \
  --anno-path split_files \
  --train-anno file_pairs_train.txt \
  --val-anno file_pairs_val.txt \
  --output-path UViT_models \
  --save-name uvit_final \
  --tensorboard \
  --batch-size 16 \
  --workers 4 \
  --sync-bn \
  --print-freq 20 \
  --k 1 \
  --epoch_block 40 \
  --num_block 3
