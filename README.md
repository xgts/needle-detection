# NDE 
## Model
<img src="https://github.com/xgts/needle-detection/blob/master/pic/framework.jpg" width="800"><br/>
- Overview of the proposed framework. The dashed box indicates a complete tracking process for a frame. We use "+" to highlight the positions in images.

<img src="https://github.com/xgts/needle-detection/blob/master/pic/network.jpg" width="800"><br/>
- Illustration of the needle detection network. The resolutions of the input and the features output by each Atrous Multi-Scale Encoding (AMSE) block are annotated. The symbol 'D' represents the dilation rate in the AMSE blocks. 'H'and 'W' represent the height and width of the video frame.

## Dataset
In the "./dataset" directory, we present a portion of the dataset used in our paper. Specifically, the content in the "./dataset/sample" directory represents a sequence of the In vitro dataset, while the content in the "./dataset/tissue" directory represents a sequence of the In vivo dataset.

## Prerequisites
- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Result
The green point is the ground truth, and the red point is the prediction result.
<img src="https://github.com/xgts/needle-detection/blob/master/pic/tissue.png" width="500"><br/>
<img src="https://github.com/xgts/needle-detection/blob/master/pic/sample.png" width="500"><br/>

## Train
- Before training, please modify the parameters in "./needle_detection_train/configs/mobilev2_mlsd_large_512_base2_bsize24.yaml".
```bash
cd needle_detection_train
python train.py
```

## Test
- In the "./Test_kal" directory, please modify the path in the main.py and load the model parameters in the Pre.py before testing.
- To test on the In vivo dataset, load "../needle_detection_train/weight_aug_kal/test/best.pth".
- To test on the In vitro dataset, load ".. /needle_detection_train/weight_sample/best.pth".
```bash
cd Test_kal
python main.py
```
