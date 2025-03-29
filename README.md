# NDE 
## Model
Overview of the proposed framework. The dashed box indicates a complete tracking process for a frame. We use "+" to highlight the positions in images.

<img src="https://github.com/xgts/needle-detection/blob/master/pic/framework.jpg" width="800" height="318"><br/>

Illustration of the needle detection network. The resolutions of the input and the features output by each Atrous Multi-Scale Encoding (AMSE) block are annotated. The symbol 'D' represents the dilation rate in the AMSE blocks. 'H'and 'W' represent the height and width of the video frame.

<img src="https://github.com/xgts/needle-detection/blob/master/pic/network.jpg" width="800" height="318"><br/>

## Dataset
In the "./dataset" directory, we present a portion of the dataset used in our paper. Specifically, the content in the "./dataset/sample" directory represents a sequence of the in-vitro dataset, while the content in the "./dataset/tissue" directory represents a sequence of the in-vivo dataset.

## Train
Before training, please modify the parameters in "mlsd_test/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml".
```bash
cd mlsd_test
python train.py
```

## Test
In the "./Test_kal" directory, please modify the path in the main.py and load the model parameters in the Pre.py, where "../mlsd_test/weight_aug_kal/test/best.pth" is used for the In-vivo dataset, ".. /mlsd_test/weight_sample/best.pth" is used for the In-vitro dataset.
```bash
cd Test_kal
python main.py
```
