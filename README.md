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

## Models evaluated are

Supervised models: Random Forest, Gradient Boosting, Logistic Regression, Neural Network
Unsupervised models: Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoder

All of these models have used in in-domain baselines, three cross-dataset directions, ten supplementary experiments, 
& eleven techniques used to improve the performance.

## Structure of Repository

cross-dataset-thesis/
├── data/                   # Links are mentioned above                   
│   ├── CICIDS2017/
│   ├── CSE-CIC-IDS2018/
│   └── UNSW_NB15/
│
├── src/
│   ├── eda/
│   │   ├── class_distribution.py    # count of benign vs attack 
│   │   ├── compare_features.py    # overlap of columns between the datasets
│   │   ├── correlation_analysis.py  # Heatmap of feature correlation
│   │   ├── descriptive_statistics.py  # mean, minimum, maximum etc for the datasets
│   │   ├── duplicate_analysis.py   # Count of the duplicate rows
│   │   ├── feature_statistics.py   # same statistics but only for the dataset CICIDS2018
│   │   ├── inspect_datasets.py   # Check to know the size, columns, labels of the datasets
│   │   ├── missing_value_analysis.py  # count of the columns that have null values
│   │   └── outlier_analysis.py   # outlier check based on IQR for 2018
│   │
│   ├── models/
│   │   ├── autoencoder.py   # PyTorch unsupervised model for reconstruction 
│   │   ├── isolation_forest.py  # Unsupervised model
│   │   ├── local_outlier_factor.py   # Unsupervised model
│   │   ├── one_class_svm.py      # Unsupervised model
│   │   └── supervised_baselines.py   # consists of 4 supervised labeled models
│   │
│   ├── preprocessing/
│   │   ├── cic_feature_alignment.py   # column matching between the two CIC datasets
│   │   ├── coral_alignment.py   # logic of coral reshaping
│   │   ├── preprocessing.py    # main cleaning of datasets & mapping of the schema
│   │   ├── test_preprocessing_cic2017.py
│   │   ├── test_preprocessing_cic2018.py
│   │   └── test_preprocessing_unsw.py
│   │
│   └── studies/
│       ├── analysis/                                                  
│       │   ├── cross_dataset_supervised_2017_source.py       
│       │   ├── cross_dataset_supervised_unsw_source.py        
│       │   ├── cross_dataset_unsupervised_2017_source.py
│       │   ├── cross_dataset_unsupervised_unsw_source.py
│       │   ├── diagnostic_error_overlap_from_2017_to_2018.py   # Analysis to check why one Autoencoder is 0
│       │   ├── in_domain_baseline_supervised_2017.py
│       │   ├── in_domain_baseline_supervised_2018.py
│       │   ├── in_domain_baseline_unsupervised_2017.py
│       │   └── in_domain_baseline_unsupervised_2018.py
│       │
│       ├── experiments/                                               
│       │   ├── exp1_saturation_leakage.py   # shows why ordered & not random sampling breaks things
│       │   ├── exp2_domain_shift_analysis.py  # Distances of Euclidean/Wasserstein 
│       │   ├── exp3_pca_visualization.py
│       │   ├── exp4_contamination_study.py  # sweeping of Isolation forest's setting of contamination
│       │   ├── exp5_feature_ablation.py   # removal of groups of features one at a time
│       │   ├── exp6_noise_robustness.py   # adding the noise of gaussian noise to the inputs
│       │   ├── exp7_threshold_analysis.py  # Threshold rules of Autoencoder for all three directions
│       │   ├── exp8_tsne_visualization.py
│       │   ├── exp9_improvement_unsupervised_reduced_features_unsw.py   # Getting rid of the columns filled with 0 for the dataset UNSW
│       │   └── exp10_ocsvm_lof_collapse_diagnostic.py    # To know why the two models failed when the source dataset is UNSW
│       │
│       └── improvements/                                              
│           ├── supervised/                                           
│           │   ├── improvement_coral_alpha_sweep.py
│           │   ├── improvement_coral_full_alignment.py
│           │   ├── improvement_divergence_feature_selection.py
│           │   ├── improvement_few_shot_finetuning_all_directions.py
│           │   └── improvement_fine_grained_label_budget_study.py    # how some labels are genuinely needed
│           │
│           └── unsupervised/                                          
│               ├── improvement_target_rescaling_unsupervised.py
│               ├── improvement_unsupervised_auto_selected_pipeline.py  # this picks the best configuration by default
│               ├── improvement_unsupervised_combined_training.py
│               ├── improvement_unsupervised_ensemble.py
│               ├── improvement_unsupervised_fewshot_calibration.py
│               └── improvement_unsupervised_stacked_meta.py
│
└── results/                                    # When you run the files above, outputs/results/plots are saved here by default
    ├── eda/
    ├── studies/
    │   ├── analysis/
    │   ├── experiments/
    │   └── improvements/
    │       ├── supervised/
    │       └── unsupervised/

## Headline Finding

**Cross dataset collapse depends on the direction and not universal**

| Direction | Best in-domain F1 | Best cross-dataset F1 (supervised, without any improvement technique) |
|---|---|---|
| UNSW-NB15 > CICIDS2017 | 0.95 | 0.354 (considerable performance have retained) |
| UNSW-NB15 > CSE-CIC-IDS2018 | 1.00 | 0.185 (considerable performance have retained) |
| CICIDS2017 > CSE-CIC-IDS2018 | 0.95 | 0.001 (almost total collapse) |

- Supervised models showed total failure when the dataset CICIDS2017 is used to train and dataset CSE-CIC-IDS2018 used to test
- When the dataset UNSW-NB15 is used in training, this total failure have not happened, though the dataset UNSW-NB15 has different toolchain from the cicids datasets.
- No similarity in the structure of the datasets doesn't mean its generalization is poor

