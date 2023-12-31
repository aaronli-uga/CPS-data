'''
Author: Qi7
Date: 2023-06-13 17:23:17
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-07-01 17:17:40
Description: 
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
# from sklearn.feature_extraction import RFECV
import xgboost as xgb

from models import ANN
from utils import model_train_detection, model_train_diagnosis
from loader import RegularLoader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

#%% feature selection comparison
def addlabels(x,y, margin):
    for i in range(len(x)):
        plt.text(i + margin, y[i], y[i])

f1_all_benchmark = [0.9722, 1, 1, 1, 1]
f1_all = [0.9833, 1, 1, 1, 1]
objects = ('KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'ANN')
y_pos = np.arange(len(objects))

plt.figure(figsize=(12, 10))
# plt.bar(y_pos, f1_all, align='center', alpha=0.5, label='proposed')
plt.bar(y_pos + 0.2, f1_all_benchmark, 0.4, label='benchmark')
plt.bar(y_pos - 0.2, f1_all, 0.4, label='proposed')
# plt.bar(y_pos + 0.2, f1_all_benchmark, 0.4, label='benchmark')
# addlabels(y_pos, f1_all, margin=-0.3, fontsize=25)
# addlabels(y_pos, f1_all_benchmark, margin=0.1)
plt.xticks(y_pos, objects, fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('Different Methods', fontsize=35)
plt.ylabel('Test Accuracy', fontsize=30)
plt.ylim((0.95, 1.01))
plt.title('Feature selection performance (proposed method vs benchmark)', fontsize=35)
# plt.title('F1 score in different levels of noise')
plt.legend(fontsize=30)


plt.show()

exit()

# test on pi1
df = pd.read_csv("dataset/systematic_data/cpu/cpu_pi1testbed_final.csv")
df = df.loc[df["cpu"] == "cpu-total"]
df = df.reset_index()
df_labels = df[['class_1', 'class_2']]

binary_class1 = df['class_1'].astype('category')
label_color = ['green' if i==1 else 'Red' for i in binary_class1]

df_features_cpu = df.drop(columns=['index', 'Unnamed: 0', '_time', 'cpu', 'host', 'class_1', 'class_2'])

df = pd.read_csv("dataset/systematic_data/diskio/diskio_pi1testbed_final.csv")
df_features_diskio = df.drop(columns=['Unnamed: 0', '_time', 'name', 'host', 'class_1', 'class_2'])

df = pd.read_csv("dataset/systematic_data/memory/mem_pi1testbed_final.csv")
df_features_mem = df.drop(columns=['Unnamed: 0', '_time', 'host', 'class_1', 'class_2'])

df = pd.read_csv("dataset/systematic_data/network/network_pi1testbed_final.csv")
df_features_network = df.drop(columns=['Unnamed: 0', '_time', 'interface', 'host', 'class_1', 'class_2'])

df = pd.read_csv("dataset/systematic_data/processes/processes_pi1testbed_final.csv")
df_features_processes = df.drop(columns=['Unnamed: 0', '_time', 'host', 'class_1', 'class_2'])

df = pd.read_csv("dataset/systematic_data/system_info/system_pi1testbed_final.csv")
df_features_system = df.drop(columns=['Unnamed: 0', '_time', 'host', 'uptime', 'uptime_format', 'class_1', 'class_2'])


#%%
class_encoder = LabelEncoder()
scaler = MinMaxScaler()
df_labels['class_1'] = class_encoder.fit_transform(df_labels['class_1'])
df_labels['class_2'] = class_encoder.fit_transform(df_labels['class_2'])
df_clean = pd.concat([df_features_cpu, df_features_diskio, df_features_mem, df_features_network, df_features_processes, df_features_system, df_labels], axis=1)
df_feature_extract_baseline = df_clean
# df_clean = df_clean[["write_bytes", "active", "buffered", "cached", "slab", "sreclaimable", "sunreclaim", "bytes_recv", "bytes_sent", "packets_recv", "packets_sent", "class_1", "class_2"]]
df_clean = df_clean[["write_bytes", "write_time", "merged_writes", "writes", "usage_system", "slab", "used", "bytes_recv", "bytes_sent", "total_threads", "io_time", "load15", "load1", "class_1", "class_2"]]



baseline = df_feature_extract_baseline.corr(method='pearson')
baseline = abs(baseline['class_1'])
baseline = baseline.dropna()
baseline = baseline[baseline > 0.5]


# dataframe for PCA : PCA can not have 'categorical' features
df_tmp = df_clean.drop(['class_1', 'class_2'], axis=1)
# Mean normalization
normalized_df = (df_tmp - df_tmp.mean()) / df_tmp.std()

# all the features
feature_names = df_tmp.columns.to_list() 

# PCA
# pca = PCA(n_components=2)
# pca.fit(normalized_df)
# df_pca = pca.transform(normalized_df)
# df_pca = pd.DataFrame(df_pca)
# df_pca.columns = ['PCA component 1', 'PCA component 2']
# df_pca.plot.scatter(x='PCA component 1', y='PCA component 2', marker='o',
#         alpha=0.6, # opacity
#         color=label_color,
#         title="red: attack, green: normal" )
# plt.show()
# %%
features = normalized_df.to_numpy()
# features = df_tmp.to_numpy()
targets =  df_clean['class_2'].to_numpy()
targets = np.expand_dims(targets, axis=1)


# split the training and test data
train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets,
        train_size=0.8,
        test_size=0.2,
        # random but same for all run, also accuracy depends on the
        # selection of data e.g. if we put 10 then accuracy will be 1.0
        # in this example
        random_state=23,
        # keep same proportion of 'target' in test and target data
        stratify=targets
    )

#%%
# use LogisticRegression
# classifier = KNeighborsClassifier()
# classifier = DecisionTreeClassifier()
# classifier = RandomForestClassifier()
classifier =xgb.XGBClassifier()
# training using 'training data'
classifier.fit(train_features, train_targets) # fit the model for training data

# predict the 'target' for 'training data'
prediction_training_targets = classifier.predict(train_features)
self_accuracy = accuracy_score(train_targets, prediction_training_targets)
print("Accuracy for training data (self accuracy):", self_accuracy)
self_f1 = f1_score(train_targets, prediction_training_targets, average='micro')
print("F1 score for training data (self f1score):", self_f1)

# predict the 'target' for 'test data'
prediction_test_targets = classifier.predict(test_features)
test_accuracy = accuracy_score(test_targets, prediction_test_targets)
print("Accuracy for test data:", test_accuracy)
test_f1 = f1_score(test_targets, prediction_test_targets, average='micro')
print("F1 score for test data:", test_f1)

# feature importances
# f_i = list(zip(feature_names, classifier.feature_importances_))
# f_i.sort(key = lambda x : x[1])
# # plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
# plt.figure(figsize=(12, 10))
# plt.barh([x[0] for x in f_i[-20:]],[x[1] for x in f_i[-20:]])
# plt.xticks(fontsize=30)
# plt.yticks(fontsize=20)
# plt.xlabel("Importance Score", size = 30)
# plt.show()

# %%
# n_components = 2
# tsne = TSNE(n_components)
# tsne_result = tsne.fit_transform(features)
# tsne_result = pd.DataFrame(tsne_result)
# tsne_result.columns = ['tsne component 1', 'tsne component 2']
# tsne_result.plot.scatter(x='tsne component 1', y='tsne component 2', marker='o',
#         alpha=0.5, # opacity
#         color=label_color,
#         title="red: attack, green: normal" )
# plt.show()



#%% ANN models detection
# save_model_path = "saved_models/"
# trainset = RegularLoader(train_features, train_targets)
# validset = RegularLoader(test_features, test_targets)

# # Hyper parameters
# batch_size = 512
# learning_rate = 0.005
# num_epochs = 500
# history = dict(val_loss=[], val_acc=[], val_f1=[], val_f1_all=[], train_loss=[], train_acc=[], train_f1=[], train_f1_all=[])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = "cpu"

# trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
# validloader = DataLoader(validset, shuffle=True, batch_size=batch_size) # get all the samples at once
# model = ANN(n_input=train_features.shape[1], n_classes=1)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# model_train_detection(
#     model=model, 
#     train_loader=trainloader, 
#     val_loader=validloader,
#     num_epochs=num_epochs,
#     optimizer=optimizer,
#     device=device,
#     history=history
# )

# torch.save(model.state_dict(), save_model_path + f"detection_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
# np.save(save_model_path + f"detection_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)

#%% ANN models diagnosis
# save_model_path = "saved_models/"
# trainset = RegularLoader(train_features, train_targets)
# validset = RegularLoader(test_features, test_targets)

# # Hyper parameters
# batch_size = 512
# learning_rate = 0.001
# num_epochs = 500
# history = dict(val_loss=[], val_acc=[], val_f1=[], val_f1_all=[], train_loss=[], train_acc=[], train_f1=[], train_f1_all=[])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = "cpu"

# trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
# validloader = DataLoader(validset, shuffle=True, batch_size=batch_size) # get all the samples at once
# model = ANN(n_input=train_features.shape[1], n_classes=4)
# model.to(device)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# model_train_diagnosis(
#     model=model, 
#     train_loader=trainloader, 
#     val_loader=validloader,
#     num_epochs=num_epochs,
#     optimizer=optimizer,
#     device=device,
#     history=history
# )

# torch.save(model.state_dict(), save_model_path + f"system_diagnosis_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
# np.save(save_model_path + f"system_diagnosis_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)




# %%
