import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

os.makedirs("results/eda/feature_statistics", exist_ok=True )
df = pd.read_csv( "data/CSE-CIC-IDS2018/02-14-2018.csv" )
statistics = df.describe()
statistics.to_csv("results/eda/feature_statistics/cic2018_statistics.csv")
print(statistics)