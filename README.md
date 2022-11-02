# Going Deeper with Convolutional Neural Network for Stock Market Prediction
Repository for Going Deeper with Convolutional Neural Network for Stock Market Prediction

## Introduction
Predict the stock market price will go up or not in the near future.

## Data Collection
- Using Yahoo! Finance for time series data source
- 50 Taiwan Companies from 0050.TW index.
- Top 10 Indonesia Stock exchange companies.

## Methodology
- Using candlestick chart for input model
- DeepCNN
- ResNet 50 - WIP
- VGG16 - WIP
- VGG19 - WIP
- Randomforest - WIP
- KNN - WIP

## Usage
### Prepare Environment
Recommended using virtual environment
```
python3 -m venv .env
```

Running on Python 3
```
pip install -U -r requirements.txt
```

### Prepare Config table
| Name | Desc | Value |
|------|------|-------|
| label | Column name for label | Label
| ticker | List of financial stock code separated by comma | e.g TLKM.JK,BBCA.JK,HMSP.JK,BBRI.JK |
| test_start | Test data start date time | e.g 2021-01-01 |
| test_end | Test data end date time | e.g 2021-05-31 |
| train_start | Train data start date time | |
| train_end | Train data end date time | e.g 2020-12-31 |
| window_size | The sliding window size of trading data | e.g 20 |
| img_size | The candlestick chart image size | e.g 128 |
| img_channel | The color channel (RGB) | e.g 3 |
| clean | Boolean to clean data | e.g True |
| epoch | Number of iteration when training the model | e.g 30 |
| batch_size | The batch size number during training model | e.g 64 |
| model_name | Model name | cnn |
| dataset | The dataset directory | dataset/ |


### Training
```
python deepcandle.py
```


## Performance Evaluation
- Accuracy
- Specitivity
- Sensitivity
- MCC
- F1

## Citation
```
@misc{1903.12258,
Author = {Rosdyana Mangir Irawan Kusuma and Trang-Thi Ho and Wei-Chun Kao and Yu-Yen Ou and Kai-Lung Hua},
Title = {Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market},
Year = {2019},
Eprint = {arXiv:1903.12258},
}
```
