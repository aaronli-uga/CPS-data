'''
Author: Qi7
Date: 2023-06-14 21:16:05
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-18 12:06:38
Description: 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
import xgboost as xgb


# Features: only consider sensor1 and sensor2. 
# sensor1_AC_mag, sensor1_AC_freq, sensor1_AC_thd, sensor1_DC_freq, sensor1_DC_mag, sensor1_DC_thd,
# sensor2_AC_mag, sensor2_AC_freq, sensor2_AC_thd, sensor2_DC_freq, sensor2_DC_mag, sensor2_DC_thd,

df = pd.read_csv("dataset/physical_data/physical_final.csv")

print(df.columns)

feature = "sensor2_AC_mag"
delta = False
# delta = True


attack_df = df.loc[df['class_1'] == 'attack']
normal_df = df.loc[df['class_1'] == 'normal']


# delta process make it better
if delta:
    normal_df_feature = np.diff(normal_df[feature].to_numpy())
    normal_df_feature = np.append(normal_df_feature, normal_df_feature[-1])
    attack_df_feature = np.diff(attack_df[feature].to_numpy())
    attack_df_feature = np.append(attack_df_feature, attack_df_feature[-1])


plt.figure(figsize=(12, 10))
if delta:
    plt.plot(normal_df["_time"], normal_df_feature, color='green', label="normal")
    plt.plot(attack_df["_time"], attack_df_feature, color='red', label='attack')
else:
    plt.plot(normal_df["_time"], normal_df[feature], color='green', label="normal")
    plt.plot(attack_df["_time"], attack_df[feature], color='red', label='attack')

# plt.xticks(np.arange(0, len(normal_df["_time"]) + len(attack_df[feature]), 2000))
plt.xticks([])
plt.yticks(fontsize=60)
if delta:
    plt.ylabel(f"{feature}_delta", size = 60)
else:
    # plt.ylabel(f"{feature}", size = 60)
    plt.ylabel("THD", size = 60)
plt.xlabel("Time", size = 60)
# Setting the number of ticks
plt.legend(fontsize=60)
plt.show()  