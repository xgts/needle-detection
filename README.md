# NDE 
### Train:
Before training, please modify the parameters in "mlsd_test/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml".
```bash
cd mlsd_test
python train.py
```

### Test:
In the Test_kal directory, please modify the path in the main.py and load the model parameters in the Pre.py, where "../mlsd_test/weight_aug_kal/best.pth" is used for the In-vivo dataset, ".. /mlsd_test/weight_sample/best.pth" is used for the In-vitro dataset.
```bash
cd Test_kal
python main.py
```
