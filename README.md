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

Naive RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.
![random_oversampling](https://user-images.githubusercontent.com/103727169/192158862-bb1749b9-6290-46e5-8ed4-d263ee9818df.png)

* Balanced accuracy score : 64%.

  ![balance_accuracy](https://user-images.githubusercontent.com/103727169/192158995-3ed8b87d-d2bd-413a-9659-4ff51324c29f.png)

* The "High Risk" precision rate was only 1% with the recall at 72% giving this model an F1 score of 2%.
* "Low Risk" had a precision rate of 100% and recall at 60%.

  ![conf_matrix](https://user-images.githubusercontent.com/103727169/192159159-2c551783-7477-4293-abf0-226dc8bb0b70.png)
  ![classifcationreport](https://user-images.githubusercontent.com/103727169/192159174-7daf5495-7901-413e-ad93-f84a65547a79.png)

SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.

* The balanced accuracy score improved slightly to 65.8%.

  ![smot_balanceaccuracy](https://user-images.githubusercontent.com/103727169/192159303-bb7bd44d-a524-4135-85c4-fcd96aebddc4.png)

* Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 62% giving this model an F1 score of 2%.

* "Low Risk" had a precision rate of 100% and an improved recall at 69%.

  ![smot_matrix](https://user-images.githubusercontent.com/103727169/192159669-184b2792-f71e-4ae5-b976-82315600a019.png)
  ![smot_Clasification](https://user-images.githubusercontent.com/103727169/192159689-3b10170d-4d63-4b86-9b19-10ef550ae4f2.png)

## Undersampling

ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

![clusteroid_counter](https://user-images.githubusercontent.com/103727169/192159857-c206ba08-97a1-43ab-b4fc-5617b8dda158.png)

* Balanced accuracy score was lower than the oversampling models at 54.5%.
  ![under_accuracy](https://user-images.githubusercontent.com/103727169/192159965-b7ec1224-bb54-403d-9f16-4d9919eb38e1.png)

* The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
* "Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.
  ![under_matriz](https://user-images.githubusercontent.com/103727169/192160050-3e9b01e4-1300-4891-a0f9-6e48e58bdd8d.png)
  ![under_classification](https://user-images.githubusercontent.com/103727169/192160063-347531c2-73ba-403e-a318-59e60d7a3396.png)


# Deliverable 2 > Use the SMOTEENN algorithm to Predict Credit Risk

## Combination (Over and Under) Sampling

SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.

![smoteenn_counter](https://user-images.githubusercontent.com/103727169/192160396-8b11e4ed-0ba4-482b-8755-17832ff969b0.png)

* The balanced accuracy score improved to 64.5% when using a combined sampling model.
  ![smoteenn_accuracy](https://user-images.githubusercontent.com/103727169/192160452-b03e98b5-01c2-429b-8551-e3b1f22832af.png)
  
* The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
* "Low Risk" still showed a precision rate of 100% with the recall at 57%.

  ![smoteenn_confusionmatrix](https://user-images.githubusercontent.com/103727169/192160476-2ebd42e6-0d15-42db-8818-1683ece72606.png)
  ![smoteenn_classificationreport](https://user-images.githubusercontent.com/103727169/192160489-c4c53e89-409b-4b18-8e0d-9cb8f8b6704d.png)








