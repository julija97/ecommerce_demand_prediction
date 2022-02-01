# FORECASTING E-COMMERCE DEMAND: A SPATIO-TEMPORAL APPROACH USING LONG SHORT-TERM MEMORY (LSTM) NETWORKS

In this project, a machine learning-based model that can predict future short-term sales demand in each area of a state based on historical data is proposed. Predictions are computed on a zone level, which is achieved via constrained K-means clustering based on  order spatial distribution. The LSTM model is trained globally making use of cross-series demand data, where each spatial zone is treated as a separate time series. The proposed method is evaluated using a dataset of e-commerce orders in Sao Paulo, Brazil. The accuracy of predictions is tested on RMSE, MAE and MSE error metrics. The method is compared with several other approaches and in order to quantify the added value of using the global model, the method is benchmarked against predictions acquired when training a local Random Forest model for each region. The application of the model to a Brazilian e-commerce dataset shows that the order arrival level can be accurately predicted 6 hours in advance.

Though Pandas is mostly used for the data pre-processing, the LSTM is implemented in [Darts](https://github.com/unit8co/darts). 

## Objective

The objective of this work is to construct a machine learning model that would be able to accurately predict demand for each spatial area based on historical data to support decision making as well as propose a systematic implementation methodology, which can be applied in route optimisation/resource allocation decision support software. Predictions are made for 6 hour periods.

It is firstly chosen to cluster the city regions based on the demand data and then predict the demand in each area using a single (LSTM) recurrent neural network (RNN) model trained on the demand patterns of all areas. With this formulation, the patterns learned by the LSTM in one area can be used in other areas. The reason why a LSTM recurrent neural network is used is because demand prediction is a time-series problem and LSTM is a state-of-the-art method for capturing long-term dependencies.

## Data

The dataset utilized in this paper has been extracted from [Kaggle](https://www.kaggle.com/olistbr/brazilian-ecommerce) provided by Olist. The data is released under the CC BY-NC-SA 4.0 license. Further information on the CC BY-NC-SA 4.0 license can be found [here](https://creativecommons.org/licenses/by-nc-sa/4.0/). It is a real Brazilian e-commerce public dataset of orders made at the Olist Store, the largest department store in Brazilian marketplaces. The dataset has information about around 100k orders from 2016 to 2018. Its features allow viewing an order from multiple dimensions: order status, price, payment, customer location, product attributes, etc. Each order is identified by an order ID. 


## Experimental setup

![Picture1](https://user-images.githubusercontent.com/76480153/150186351-5f3d2087-1a41-4be1-934c-9ce4ff638049.png)

## Data Exploration

<img width="453" alt="Picture2" src="https://user-images.githubusercontent.com/76480153/150408687-7eba787a-7037-4ec0-92da-1cfe19976aa2.png">

After plotting the data on the map, it can be seen that a few data points fall outside of Sao Paolo area.

<img width="478" alt="Picture3" src="https://user-images.githubusercontent.com/76480153/150408940-5862c9e7-4074-4a72-8c4b-12d4cab71afd.png">

After plotting the data's temporal patterns, an outlier is seen on November 24th. This can be attributed to the Black Friday Sales in Brazil. Furthermore, an increasing trend in data is seen. Lastly, the beginning and the end of series have very sparse records.

It is important to identify certain seasonality patterns present in the data to enhance predictive performance of the model by adding them as features.

<img width="452" alt="Picture4" src="https://user-images.githubusercontent.com/76480153/150429043-d53796a0-c257-48fa-9571-c5b5282d667d.png">

It can be observed that on the weekend, especially Saturday, the order count purchased is slightly smaller than on the other days while Monday holds the biggest share of orders.

<img width="452" alt="Picture5" src="https://user-images.githubusercontent.com/76480153/150429123-33c27473-49f6-4822-b805-36c75851b4d9.png">

It can be seen that order count decreases dramatically between the hours 0.00 and 10.00 with the rest of the period in the day following relatively even patterns.

Summarizing the findings of the analysis, a weak impact of the day of the week and the time of the day on the order placement can be identified. 

## Data Pre-Processing

### Data Cleaning, Aggregation 

In the collected data, there are many attributes recorded, however, not all of them are useful for model creation since many of them are not informative. After cleaning the unnecessary features, all of the records that are not from the Sao Paulo state (SP) are discarded. This is because this study focuses at providing predictions for a mega-city and its surrounding areas. 

In order to have the spatial locations of each customer order, the longitude and latitude of each transaction have to be recorded. While each order has its spatial dimension recorded at a zip code prefix level, the geolocation data set, however, contains multiple coordinate values for each zip code prefix. The mean values of all longitude and latitude values are computed for each zip code prefix in order to obtain one optimal value for each.

In order to obtain one common data frame used for the analysis, the geolocation, customers and orders data frames have to be merged. The orders data frame is merged with the customer data frame by a unique identifier “customer_id”, while the geolocation data frame is additionally merged by a “zip_code_prefix” resulting in a unified data frame depicted in Figure 

![Picture7](https://user-images.githubusercontent.com/76480153/150443111-9493fcfa-fdcc-49b7-88a6-40b565a20574.png)

Data from the year 2016 as well as September and October of the year 2018 is removed due to sparsity. Furthermore, As the “Black Friday” will not recur in the forecasting period, the values on that day are therefore replaced with NaN values in order to allow for linear interpopulation. Lastly, 5 points lying outside of Sao Paulo are discarded as outliers.

<img width="380" alt="Picture8" src="https://user-images.githubusercontent.com/76480153/150443936-70f35bf6-bf95-4b6d-94af-3e7897aaa75c.png">

## Zoning

Clustering  is chosen to produce geographically heterogeneous spatial clusters based on order location (latitude and longitude coordinates) and associated order volume densities.

In order to avoid some locations having no demand at multiple time periods, complicating the prediction task, a [K-means constrained package](https://pypi.org/project/k-means-constrained/) is employed to group the data points into clusters based on certain thresholds. 

Using the merged orders dataset, the latitude and longitude coordinates for each order were extracted and processed through a constrained K-Means clustering package specifying 8000 as the minimum number of points and 9000 as the maximum number of points that fall into each cluster. 


![Picture9](https://user-images.githubusercontent.com/76480153/150444133-d94bba9d-6851-457e-8a08-ed295aab3f0c.png)

 ## Time Binning
 
It is decided to produce forecasts every 6 hours each time bin representing a period in a day: morning (06.00a.m.-12p.m), afternoon (12 p.m-06p.m.), evening (06p.m.-12a.m.) and night (12a.m-06a.m). Order counts are therefore grouped to specific time intervals.

<img width="291" alt="Picture10" src="https://user-images.githubusercontent.com/76480153/150444622-3abff261-c854-4219-b71f-774eedc2e08f.png">

Time series objects in Darts are created to be able to perform predictions in Darts.

## Training and Testing Split

80% of data is retained for training, 10% for validation and 10% for testing.

## 	Normalisation and feature encoding

As both features used in this analysis are of cyclical nature, all of them in the training, validation and testing data are encoded with cyclical encoding.

Min-max scaling method is used to ensure all values in training, validation and testing datasets lie within a fixed range [0,1]. 

## Hyperparameter Tuning

Trial-error was used to perform hyperparameter tuning. The tested combinations are illustrated.

<img width="393" alt="Picture13" src="https://user-images.githubusercontent.com/76480153/150446166-97c7472b-3513-4792-a954-95eb9b5f759e.png">


<img width="452" alt="Picture11" src="https://user-images.githubusercontent.com/76480153/150445833-7d839351-0138-460b-8daa-80db166ab1cd.png">

## Evaluation

Sample illustration of model's performance to make 6 hour demand predictions in Zone 2. This is based on testing dataset.

![Picture14](https://user-images.githubusercontent.com/76480153/150446392-020db55f-d9bd-4148-996a-59493d5252a9.png)

Proposed model was benchmarked against a locally and globally trained Random Forest model.

## Summary

This method proves to make use of data available in other areas and therefore can be especially useful in settings where enough training data is not available to a specific location. It has also been found that state-of-the-art models as LSTM do outperform simpler machine learning models in time-series forecasting. 

## Future Work

- A deeper model with more data available
- Multiple timestep predictions

## Updates:

Code is still a work in progress









