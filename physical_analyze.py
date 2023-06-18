'''
Author: Qi7
Date: 2023-04-04 19:53:35
LastEditors: aaronli-uga ql61608@uga.edu
LastEditTime: 2023-06-17 23:29:32
Description: 
'''
#%%
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

from models import ANN
from utils import model_train_detection, model_train_diagnosis
from loader import RegularLoader
import torch
import torch.optim as optim
from torch.utils.data import DataLoader


df = pd.read_csv("dataset/physical_data/physical_final.csv")


#%%
# Drop the unrelated protocols

class_encoder = LabelEncoder()
scaler = MinMaxScaler()
df['class_1'] = class_encoder.fit_transform(df['class_1'])
df['class_2'] = class_encoder.fit_transform(df['class_2'])
df_clean = df[['sensor1_AC_freq', 'sensor1_AC_mag', 'sensor1_AC_thd', 'sensor1_DC_freq', 'sensor1_DC_mag', 'sensor1_DC_thd', 'sensor2_AC_freq', 'sensor2_AC_mag', 'sensor2_AC_thd', 'sensor2_DC_freq', 'sensor2_DC_mag', 'sensor2_DC_thd', 'class_1', 'class_2']]

binary_class1 = df['class_2'].astype('category')
label_color = ['#2ca02c' if i==1 else '#d62728' for i in binary_class1]

# dataframe for PCA : PCA can not have 'categorical' features
df_tmp = df_clean.drop(['class_1', 'class_2'], axis=1)
# Mean normalization
normalized_df = (df_tmp - df_tmp.mean()) / df_tmp.std()

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
features = normalized_df.to_numpy()
targets =  df_clean['class_1'].to_numpy()
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

# %% machine learning methods
# # classifier = KNeighborsClassifier()
# # classifier = DecisionTreeClassifier()
# # classifier = RandomForestClassifier()
# # classifier =xgb.XGBClassifier()

# # training using 'training data'
# classifier.fit(train_features, train_targets) # fit the model for training data

# # predict the 'target' for 'training data'
# prediction_training_targets = classifier.predict(train_features)
# self_accuracy = accuracy_score(train_targets, prediction_training_targets)
# print("Accuracy for training data (self accuracy):", self_accuracy)

# # predict the 'target' for 'test data'
# prediction_test_targets = classifier.predict(test_features)
# test_accuracy = accuracy_score(test_targets, prediction_test_targets)
# print("Accuracy for test data:", test_accuracy)
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
# plt.legend(labels=label_color)
# plt.show()


# %% ANN models detection


save_model_path = "saved_models/"
trainset = RegularLoader(train_features, train_targets)
validset = RegularLoader(test_features, test_targets)

# Hyper parameters
batch_size = 512
learning_rate = 0.005
num_epochs = 500
history = dict(val_loss=[], val_acc=[], val_f1=[], val_f1_all=[], train_loss=[], train_acc=[], train_f1=[], train_f1_all=[])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
validloader = DataLoader(validset, shuffle=True, batch_size=batch_size) # get all the samples at once
model = ANN(n_input=train_features.shape[1], n_classes=1)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

model_train_detection(
    model=model, 
    train_loader=trainloader, 
    val_loader=validloader,
    num_epochs=num_epochs,
    optimizer=optimizer,
    device=device,
    history=history
)

torch.save(model.state_dict(), save_model_path + f"detection_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
np.save(save_model_path + f"detection_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)

#%% ANN models diagnosis
# save_model_path = "saved_models/"
# trainset = RegularLoader(train_features, train_targets)
# validset = RegularLoader(test_features, test_targets)

# # Hyper parameters
# batch_size = 512
# learning_rate = 0.0002
# num_epochs = 2
# history = dict(val_loss=[], val_acc=[], val_f1=[], val_f1_all=[], train_loss=[], train_acc=[], train_f1=[], train_f1_all=[])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = "cpu"

# trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
# validloader = DataLoader(validset, shuffle=True, batch_size=batch_size) # get all the samples at once
# model = ANN(n_input=features.shape[1], n_classes=4)
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

# torch.save(model.state_dict(), save_model_path + f"detection_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_best_model.pth")
# np.save(save_model_path + f"detection_epochs{num_epochs}_lr_{learning_rate}_bs_{batch_size}_history.npy", history)