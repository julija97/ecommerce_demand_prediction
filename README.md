# FORECASTING E-COMMERCE DEMAND: A SPATIO-TEMPORAL APPROACH USING LONG SHORT-TERM MEMORY (LSTM) NETWORKS

In this project, a machine learning-based model that can predict future short-term sales demand in each area of a state based on historical data is proposed. Predictions are computed on a zone level, which is achieved via constrained K-means clustering based on  order spatial distribution. The LSTM model is trained globally making use of cross-series demand data, where each spatial zone is treated as a separate time series. The proposed method is evaluated using a dataset of e-commerce orders in Sao Paulo, Brazil. The accuracy of predictions is tested on RMSE, MAE and MSE error metrics. The method is compared with several other approaches and in order to quantify the added value of using the global model, the method is benchmarked against predictions acquired when training a local Random Forest model for each region. The application of the model to a Brazilian e-commerce dataset shows that the order arrival level can be accurately predicted 6 hours in advance.

Though Pandas is mostly used for the data pre-processing, the LSTM is implemented in Darts, a new library for user-friendly time series manipulation. 


## Experimental setup: deployment of the predictive methodology 

![Picture1](https://user-images.githubusercontent.com/76480153/150186351-5f3d2087-1a41-4be1-934c-9ce4ff638049.png)
