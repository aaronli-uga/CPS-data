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


# Protocols: ARP, DNS, HTTP(?), SNMP

df = pd.read_csv("dataset/cyber_data/cyber_final.csv")

print(df.columns)
print(np.unique(df["Protocol"]))

protocol = "SNMP"
feature = "Traffic"
delta = False
# delta = True


attack_df = df.loc[(df['class_1'] == 'attack') & (df['Protocol'] == protocol)]
normal_df = df.loc[(df['class_1'] == 'normal') & (df['Protocol'] == protocol)]


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

plt.xticks(np.arange(0, len(normal_df["_time"]) + len(attack_df[feature]), 200))
plt.yticks(fontsize=40)
if delta:
    plt.ylabel(f"{feature}_delta", size = 40)
else:
    plt.ylabel(f"{feature}", size = 40)
plt.xlabel("Time", size = 20)
# Setting the number of ticks
plt.legend(fontsize=40)
plt.show()  