'''
Author: Qi7
Date: 2023-06-13 20:08:41
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-14 01:27:13
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
# import xgboost as xgb



#%% memory related. active, available(***), available_percent, buffered
df = pd.read_csv("dataset/systematic_data/memory/mem_pi1testbed_final.csv")
print(df.columns)
feature = "buffered"

attack_df = df.loc[df['class_1'] == 'attack']
normal_df = df.loc[df['class_1'] == 'normal']


# delta process make it better
# normal_df_feature = np.diff(normal_df[feature].to_numpy())
# normal_df_feature = np.append(normal_df_feature, normal_df_feature[-1])
# attack_df_feature = np.diff(attack_df[feature].to_numpy())
# attack_df_feature = np.append(attack_df_feature, attack_df_feature[-1])


plt.figure(figsize=(6, 6))
plt.plot(normal_df["_time"], normal_df[feature], color='green', label="normal")
plt.plot(attack_df["_time"], attack_df[feature], color='red', label='attack')
# plt.plot(normal_df["_time"], normal_df_feature, color='green', label="normal")
# plt.plot(attack_df["_time"], attack_df_feature, color='red', label='attack')
plt.xticks(np.arange(0, len(normal_df["_time"]) + len(attack_df[feature]), 200))
plt.yticks(fontsize=40)
plt.ylabel(f"{feature}", size = 40)
plt.xlabel("Time", size = 20)
# Setting the number of ticks
plt.legend(fontsize=40)
plt.show()  



#%% CPU related. usage_system and usage_user of cpu-total
# df = pd.read_csv("dataset/systematic_data/cpu/cpu_pi1testbed_final.csv")


# attack_df = df.loc[(df['class_1'] == 'attack') & (df["cpu"] == 'cpu-total')]
# normal_df = df.loc[(df['class_1'] == 'normal') & (df["cpu"] == 'cpu-total')]

# # attack_df = df.loc[(df['class_1'] == 'attack') & (df["cpu"] == 'cpu0')]
# # normal_df = df.loc[(df['class_1'] == 'normal') & (df["cpu"] == 'cpu0')]

# plt.figure(figsize=(12, 10))
# plt.plot(normal_df["_time"], normal_df["usage_system"], color='green', label="normal")
# plt.plot(attack_df["_time"], attack_df["usage_system"], color='red', label='attack')
# plt.xticks(np.arange(0, len(normal_df["_time"]) + len(attack_df["usage_system"]), 150))
# plt.yticks(fontsize=40)
# plt.ylabel("CPU total system usage %", size = 40)
# plt.xlabel("Time", size = 20)
# # Setting the number of ticks
# plt.legend(fontsize=40)
# plt.show()  

#%% diskio related. io_time (climbing), merged_writes, weighted_io_time, write_bytes, write_time, writes
# df = pd.read_csv("dataset/systematic_data/diskio/diskio_pi1testbed_final.csv")
# print(df.columns)
# feature = "write_bytes"

# attack_df = df.loc[df['class_1'] == 'attack']
# normal_df = df.loc[df['class_1'] == 'normal']


# # delta process make it better
# # normal_df_feature = np.diff(normal_df[feature].to_numpy())
# # normal_df_feature = np.append(normal_df_feature, normal_df_feature[-1])
# # attack_df_feature = np.diff(attack_df[feature].to_numpy())
# # attack_df_feature = np.append(attack_df_feature, attack_df_feature[-1])


# plt.figure(figsize=(6, 6))
# plt.plot(normal_df["_time"], normal_df[feature], color='green', label="normal")
# plt.plot(attack_df["_time"], attack_df[feature], color='red', label='attack')
# # plt.plot(normal_df["_time"], normal_df_feature, color='green', label="normal")
# # plt.plot(attack_df["_time"], attack_df_feature, color='red', label='attack')
# plt.xticks(np.arange(0, len(normal_df["_time"]) + len(attack_df[feature]), 200))
# plt.yticks(fontsize=40)
# plt.ylabel(f"{feature}", size = 40)
# plt.xlabel("Time", size = 20)
# # Setting the number of ticks
# plt.legend(fontsize=40)
# plt.show()  