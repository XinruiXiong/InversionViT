# 1 GPU:
# python vit_train.py \
#   --dataset flatvel-a \
#   --train-anno flatvel_a_train_vit_full.txt \
#   --val-anno flatvel_a_val_vit_full.txt \
#   --anno-path split_files \
#   --output-path vit_output \
#   --batch-size 16 \


# multiple GPU:

# full flatvelA: 
# torchrun --nproc_per_node=8 vit_train_ddp.py -ds flatvel-a -t flatvel_a_train_vit_full.txt -v flatvel_a_val_vit_full.txt --tensorboard --batch-size 32 --epochs 50

# subset flatvelA: 
torchrun --nproc_per_node=8 vit_train_ddp.py -ds flatvel-a -t flatvel_a_train_vit.txt -v flatvel_a_val_vit.txt --tensorboard --batch-size 16