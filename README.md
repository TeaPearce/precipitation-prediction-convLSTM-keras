# CIKM AnalytiCup 2017 challenge entry

## Intro

Rough unedited code used in data science contest - CIKM AnalytiCup 2017 challenge - to predict short-term rainfall. 

This solution used a convolutional recurrent neural network (convLSTM) analysing sequential radar maps. Maps were provided at 4 different altitudes, and a separate model was built for each, with predictions averaged.

Solution was inspired by the paper [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)


Final score (RMSE) was 14.8 (top 100), relative to top 5 accuracies ranging 11.0 to 13.3.

## Data wrangling

Data was provided in text file format. The first part of the program processes and stores this as .h5 file systems for ease of loading and analysing (data was too large to fit into memory).

Missing data points were simply assigned the median value of that image.

## Competition details

[Full details here](https://tianchi.aliyun.com/competition/information.htm?raceId=231596&_lang=en_US)

"Short-term precipitation forecasting such as rainfall prediction is a task to predict a short-term rainfall amount based on current observations. It has been a very important problem in the field of meteorological service. An accurate weather prediction service can support casual usages such as outdoor activity and even provide early warnings of floods or traffic accidents. To predict short-term rainfall amount, we usually make use of radar data, rain gauge data and numerical weather outputs. In this challenge, we focus on the first type of data - radar data, more specifically radar echo extrapolation data. Our target is to build a rainfall prediction model by solely using the radar data."



