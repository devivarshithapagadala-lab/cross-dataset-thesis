# Direction-Dependent Generalization in Cross-Dataset Network Intrusion Detection:
**A Comparative Study of In-Domain and Cross-Dataset Performance for Supervised and Unsupervised Machine Learning Models**

**Author:** Devi Varshitha Pagadala
**Course:** MSc Data Science, AI, Digital business
**Institution:** Gisma University of Applied Sciences
**Supervisor:** Prof. Amirhossein Jamalian

## Overview

Network Intrusion Detection Systems (NIDS) based on machine learning usually report 
almost perfect accuracy when trained & tested on the same dataset. 
This dissertation performs an investigation and evaluation on what happens when a model 
trained on a dataset is deployed to detect intrusions in a different network traffic dataset.

## Datasets 

These datasets are not included in the Github repository due to the size of the datasets,
All of these datasets are available at the given urls

**UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
**CICIDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
**CSE-CIC-IDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html

All the models in this project are only trained on old dataset and tested on new dataset, 
which has led to three directions of training and testing as below:

1. UNSW-NB15 > CICIDS2017
3. UNSW-NB15 > CSE-CIC-IDS2018
2. CICIDS2017 > CSE-CIC-IDS2018

## models evaluated are

Supervised models: Random Forest, Gradient Boosting, Logistic Regression, Neural Network
Unsupervised models: Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoder

All of these models have used in in-domain baselines, three cross-dataset directions, ten supplementary experiments, 
& eleven techniques used to improve the performance.

## Headline Finding


