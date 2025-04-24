# python vit_test.py \
#   --model InversionViT \
#   --resume ./kaggle_samples_output_v2/best_model.pth \
#   --dataset flatvel-a \
#   --val-anno ./split_files/flatvel_a_test_vit.txt \
#   --anno-path split_files \
#   --output-path vit_test_output \
#   --batch-size 16 \
#   --device cuda \
#   --k 1 \
#   --vis --vis-suffix vit_test \
#   -vb 2 -vsa 2

python vit_test.py \
  --model UViT \
  --resume ./UViT_models/uvit_final/checkpoint.pth \
  --dataset flatvel-a \
  --val-anno ./split_files/flatvel_a_test_vit.txt \
  --anno-path split_files \
  --output-path uvit_test_output \
  --batch-size 8 \
  --device cuda \
  --k 1 \
  --vis --vis-suffix uvit_test \
  -vb 2 -vsa 2