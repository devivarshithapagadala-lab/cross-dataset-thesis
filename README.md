**Cross-Dataset Generalization for Network Intrusion Detection Using Supervised 
and Unsupervised Machine Learning Models: An Evaluation of Adaptation Techniques**

This is Master's project for dissertation which involves the investigation of whether 
supervised & unsupervised machine learning models for network intrusion detection
generalize across different datasets which have specific independent data, and also 
discusses about the techniques of evaluations used where it fails, which helps in improvement of generalization

**Author:** Devi Varshitha Pagadala
**Institution:** Gisma University of Applied Sciences, Department of Computer and Data Sciences
**Supervisor:** Prof. Amirhossein Jamalian

# Overview
Network Intrusion Detection Systems (NIDS) which are based on machine learning usually
states almost perfect accuracy when trained & tested on the same dataset. This thesis 
expands this scope further by investigating a meaningful question as below-
What will happen when model which is trained on specific dataset is deployed to test on 
another network traffic data?

This is done using 3 benchmark datasets:
**CICIDS2017**
**CSE-CIC-IDS2018**
**UNSW-NB15**
which are used in:
1. Building a pipeline for feature harmonization which aligns all 3 datasets into
   a common schema with 31 features
2. Training & evaluating 8 models which are combination of four supervised models, 
   four unsupervised models across four directions of cross dataset training & testing 
   in additional to in domain baselines for both the datasets CICIDS2017 & CSE-CIC-IDS2018
3. Investigating why certain models are failing when tested under shift the domain, through 14 plus
   experiments as supplementary
4. Evaluating the ten improvement techniques which are combination of four supervised techniques, 
   six unsupervised techniques to recover the performance of lost cross dataset

# Finding headline
Supervised models (Random Forest, Gradient Boosting, Logistic Regression, Neural Network)
In domain recall is 90-99% 
Cross dataset recall (before applying improvement techniques) = ~0% (almost total collapse for every model in every direction) 
Cross dataset recall (after applying improvement techniques) = 75-98.5%, through few shot fine tuning (1-10% labeled target data)

Unsupervised models (Isolation Forest, One-Class SVM, LocalOutlierFactor, Autoencoder)
In domain recall is 64-99% 
Cross dataset recall (before applying improvement techniques) = 15-98% (the performance is weakened, but it still detects real attacks)
Cross dataset F1 (after applying improvement techniques) = 0.34-0.65, through a selected calibration combination of techniques


Supervised models learn to recognize the specific attacks they were shown during training.
so, when they see a completely different network with different attacks, they fail almost entirely, 
no matter which algorithm you use. 

Unsupervised models work differently: they never learn what attacks look like at all,
they only learn what normal traffic looks like, so they don't depend on having seen a matching attack before. 
This is why they hold up much better on a new dataset.
And in both cases, giving the model even a small number of labeled examples from the new dataset is enough 
to recover most of the performance that was lost.

# Structure of the repository
```
cross-dataset-thesis/
├── data/                                      # Raw datasets but not included in this repo due to size issues
│   ├── CICIDS2017/                              (but see below to download these datasets and load them)
│   ├── CSE-CIC-IDS2018/
│   └── UNSW_NB15/
│
├── src/
│   ├── eda/
│   │   ├── class_distribution.py                # Makes a chart showing how many normal vs. attack rows there are
│   │   ├── compare_features.py                  # Checks which column names match across the datasets
│   │   ├── correlation_analysis.py              # Makes a chart showing which features move together
│   │   ├── descriptive_statistics.py            # Prints basic numbers like average, min, and max for all datasets
│   │   ├── duplicate_analysis.py                # Counts how many rows are exact copies of another row
│   │   ├── feature_statistics.py                # Same basic stats, just for the 2018 dataset
│   │   ├── inspect_datasets.py                  # Prints size, columns, and label counts for all datasets
│   │   ├── missing_value_analysis.py            # Counts how many blank/missing values each column has
│   │   └── outlier_analysis.py                  # Finds unusually high or low values in the 2018 dataset
│   │
│   ├── models/
│   │   ├── autoencoder.py                       # A neural network that learns what normal traffic looks like
│   │   ├── isolation_forest.py                  # An anomaly-detection model built into scikit-learn
│   │   ├── local_outlier_factor.py              # Another anomaly-detection model built into scikit-learn
│   │   ├── one_class_svm.py                     # Another anomaly-detection model built into scikit-learn
│   │   └── supervised_baselines.py              # The 4 models that learn from labeled attack examples
│   │
│   ├── preprocessing/
│   │   ├── cic_feature_alignment.py             # Checks which column names are shared between the two CIC datasets
│   │   ├── coral_alignment.py                   # Reshapes data so it looks more like the target dataset
│   │   ├── preprocessing.py                     # The main script that cleans and prepares all the data
│   │   ├── test_preprocessing_cic2017.py        # Checks that cleaning CICIDS2017 works correctly
│   │   ├── test_preprocessing_cic2018.py        # Checks that cleaning CSE-CIC-IDS2018 works correctly
│   │   └── test_preprocessing_unsw.py           # Checks that cleaning UNSW-NB15 works correctly
│   │
│   └── studies/
│       ├── analysis/
│       │   ├── cross_dataset_supervised.py              # Trains on 2017 data, tests on 2018 and UNSW data
│       │   ├── cross_dataset_unsupervised.py            # Same thing, but for the models that don't use labels
│       │   ├── diagnostic_error_overlap.py              # Looks into why one result came out at zero
│       │   ├── in_domain_baseline_supervised_2017.py    # Trains and tests only on 2017 data, no dataset switch
│       │   ├── in_domain_baseline_supervised_2018.py    # Same as above, but using 2018 data instead
│       │   ├── in_domain_baseline_unsupervised_2017.py  # Same in-domain test, for the no-label models
│       │   ├── in_domain_baseline_unsupervised_2018.py  # Same as above, but using 2018 data instead
│       │   ├── reverse_cross_dataset_supervised.py      # Trains on 2018 data, tests on 2017 and UNSW data
│       │   └── reverse_cross_dataset_unsupervised.py    # Same thing, but for the no-label models
│       │
│       ├── experiments/
│       │   ├── exp1_saturation_leakage.py               # Shows why you shouldn't pick test data in order
│       │   ├── exp2_balanced_close_domain.py            # An early version of the 2017-to-2018 test
│       │   ├── exp3_heterogeneous_unsw.py               # An early version of the 2017-to-UNSW test
│       │   ├── exp4_domain_shift_analysis.py            # Measures how different the datasets are from each other
│       │   ├── exp4_pca_visualization.py                # Draws a simple 2D picture of the data
│       │   ├── exp4_tsne_visualization.py               # Draws another kind of 2D picture of the data
│       │   ├── exp5_contamination_study.py              # Tests different settings for how many attacks to expect
│       │   ├── exp6_feature_ablation.py                 # Tests which features matter most by removing them
│       │   ├── exp7_noise_robustness.py                 # Tests how well the models handle messy data
│       │   ├── exp8_threshold_analysis.py               # Tests different cutoff points for the Autoencoder
│       │   ├── exp9_reverse_threshold_analysis.py       # Same test, but in the opposite direction
│       │   ├── exp10_threshold_unsw_forward.py          # Same test, but with UNSW as the target
│       │   ├── exp11_threshold_unsw_reverse.py          # Same test, opposite direction, with UNSW as the target
│       │   ├── exp12_reduced_feature_unsw_study.py      # Tests removing UNSW's fake columns (it didn't help but kept as finding)
│       │   ├── exp13_duplicate_sensitivity.py           # Tests if removing duplicate rows changes the results
│       │   ├── exp14_extended_all_directions.py         # Same duplicate test, across all 4 directions
│       │   └── exp14b_cross_dataset_dedup_full.py       # Same as above, now including the labeled models too
│       │
│       └── improvements/
│           ├── supervised/
│           │   ├── improvement_coral_alpha_sweep.py                 # Tests different strengths of data reshaping
│           │   ├── improvement_coral_supervised.py                  # Tests full-strength data reshaping
│           │   ├── improvement_divergence_feature_selection.py      # Tests dropping the most different-looking features
│           │   └── improvement_few_shot_finetuning_all_directions.py # Tests adding a few real labeled examples during training
│           │
│           └── unsupervised/
│               ├── improvement_target_rescaling_unsupervised.py       # Tests rescaling the data using the new dataset's own numbers
│               ├── improvement_unsupervised_combined_training.py      # Tests adding some normal traffic from the new dataset
│               ├── improvement_unsupervised_ensemble.py               # Tests combining all 4 models guesses together
│               ├── improvement_unsupervised_fewshot_calibration.py    # Tests picking the best settings using a few labeled examples
│               ├── improvement_unsupervised_final_pipeline.py         # Automatically picks whichever technique works best
│               └── improvement_unsupervised_stacked_meta.py           # Tests learning the best way to combine the 4 models
│
└── results/                                    # All the output files and charts, created when you run a script
    ├── eda/
    ├── visualizations/
    └── studies/
        ├── analysis/
        ├── experiments/
        └── improvements/
            ├── supervised/
            └── unsupervised/
```

# Datasets

Not included due to size issue. Download & place them under `data/`:

- **CICIDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html
- **CSE-CIC-IDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset


# Setting up the project
# System requirements

Python version = 3.9 or higher
Operating System = macOS, Linux, or Windows
RAM = 8 GB minimum recommended
Disk space = ~5 GB free for the three raw datasets

# Step-by-step setup

```
git clone <https://github.com/Varshithapagadala/Master-thesis>
cd cross-dataset-thesis

python3 -m venv venv
source venv/bin/activate        # for macOS/Linux
venv\Scripts\activate           # for windows

pip3 install -r requirements.txt

# Test the setup by running any file in below format
python3 -m src.preprocessing.test_preprocessing_cic2017
```

# `requirements.txt`

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
torch>=2.0.0
scipy>=1.9.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

If you run into a package resolution issue on macOS use before line to fix it
`pip3 install -r requirements.txt --break-system-packages`

## Run the project like below

Preprocessing — verify the pipeline works
python3 -m src.preprocessing.test_preprocessing_cic2017
Run the other files the same way  (see the preprocessing file list above) 

EDA — exploratory data analysis
python3 -m src.eda.inspect_datasets
Run any other file in src/eda/ the same way (see the file list above)

Core analysis — baseline and cross-dataset evaluation
python3 -m src.studies.analysis.cross_dataset_supervised
Run any other file in src/studies/analysis/ the same way (see the file list above)

Supplementary experiments
python3 -m src.studies.experiments.exp1_saturation_leakage
Run exp2 to exp14 the same way (see the file list above)

Improvement techniques — supervised
python3 -m src.studies.improvements.supervised.improvement_few_shot_finetuning_all_directions
Run the other 3 files in this folder the same way (see the file list above)

Improvement techniques — unsupervised
python3 -m src.studies.improvements.unsupervised.improvement_unsupervised_final_pipeline
Run the other 5 files in this folder the same way (see the file list above)

All results are written in a way to save them in `results/` folder.

## Evaluated models
Supervised models - Random Forest, Gradient Boosting, Logistic Regression, Neural Network (MLP)
Unsupervised models - Isolation Forest, One-Class SVM, Local Outlier Factor, Autoencoder (PyTorch)

## Improvement techniques evaluated
Coral domain adaptation = Supervised, no target labels required
Feature selection with the guidance of Domain divergence= Supervised, uses small split of calibration
Few shot finetuning = Supervised, requires 1-10% of labeled target data
Rescaling on Target domain  = Unsupervised, no target labels required
Combined domain training = Unsupervised, requires benign only target data and no attack labels
Few shot hyperparameter/threshold calibration = Unsupervised, requires 1-10% of labeled target data
Ensemble score combination = Unsupervised, requires 1-10% of labeled target data
Stacked meta classifier = Unsupervised, requires 1-10% of labeled target data
Auto selected pipeline = Unsupervised, requires 1-10% of labeled target data

## Reproducibility
All scikit-learn models use `random_state=42`. 
The PyTorch Autoencoder additionally seeds Python's random, NumPy, and PyTorch (including CUDA)
explicitly in its constructor, since NumPy's seed alone does not control
internal weight initialization of PyTorch (see `src/models/autoencoder.py`).

## License
This project is submitted as part of an academic dissertation and is
provided for reference and reproducibility purposes.