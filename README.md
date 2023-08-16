# Gesture-Recognition
## Getting Started
Go to this [Google Drive](https://drive.google.com/drive/folders/1bHKrLNwaPi_dW1bedSoxD5SJKa-y-E1U?usp=sharing), and

- Download the pretrained model weights `resn152_epoch_x.pth` and place the file under `/checkpoints`
- Download Dataset files `train_xd_csv` and `test_xd_csv`



## Data Preprocessing
- We preprocess the dataset we originally collected; `train_xd_csv` and `test_xd_csv` are the dataset after preprocessing.
```
python preprocessing/preprocessing.py
```



## Train

We used two models to train:
- (Option 1)train with resnet152 
```
python train/train_resnet.py
```
- (Option 2)train with regnet_y_32g
```
python train/train_regnet.py
```


## Results
- Plot the confusion matrix for the predicted results:
  
```
python scripts/confused.py
```
