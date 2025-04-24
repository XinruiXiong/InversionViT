# InversionViT: A Vision Transformer implemention of InversionNet

## Data
https://openfwi-lanl.github.io/docs/data.html#vel

Put in the ./data 


## Run
### Train (a subset of FlatVel-A)
```bash
bash ./vit_train.sh
```
Multiple GPU training is supported.

### Test
```bash
bash ./vit_test.sh
```

### Report

https://github.com/XinruiXiong/InversionViT/blob/main/InversionViT%20Report.pdf


## Reference:
[1] Yue Wu and Youzuo Lin, “InversionNet: An Eficient and Accurate Data-driven Full Waveform Inversion,” IEEE Transactions on Computational Imaging (IF:4.5), 6(1):419-433, 2019.

[2] OpenFWI Website: https://smileunc.github.io/projects/openfwi/resources


# For Kaggle Competetion model, see *kaggle* branch
