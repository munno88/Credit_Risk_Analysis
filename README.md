# CREDIT RISK ANALYSIS
## Machine Learning
![machine-learning-hero](https://user-images.githubusercontent.com/103727169/192156117-e47bb346-9386-4cf8-828c-190006e8a5ed.jpg)

An analysis using Machine Learning algorithms to identify credit card risk using a dataset from LendingClub.

# Overview

The purpose of this analysis is to understand how to utilize Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. This reason why this is called "Supervised Learning" is because the data includes a labeled outcome.

To complete this analysis, we use different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from the LendingClub has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, we needed to employ various Machine Learning algorithms to resample the data. These algorithms include RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier, and EasyEnsembleClassifier.

# Results

As mentioned in the overview, we use Machine Learning to resample the dataset using Python libraries: scikit-learn and imbalanced-learn evaluate the results and provide a comparison for our analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

![Screenshot 2022-09-25 101407](https://user-images.githubusercontent.com/103727169/192156352-541c34bd-67ef-4204-a2a1-07388cd997f1.png)

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.


![training_set](https://user-images.githubusercontent.com/103727169/192156563-d7e7f34a-b811-4a80-9610-c2593d612c0d.png)

# Delivarable 1 > Use Resampling Models to Predict Credit Risk

## Oversampling


