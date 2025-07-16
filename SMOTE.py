import torch
import random
import math
import pandas as pd
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, ParameterSampler, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import roman


# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

dataset_label = pd.read_csv('/data/fshan/fs/songhua/All_data/Processed8/cycle_label2.csv', header= 0, index_col= 0)
dataset_unlabel = pd.read_csv('/data/fshan/fs/songhua/All_data/Processed8/cycle_unlabel2.csv', header= 0, index_col= 0)
input_label = dataset_label.iloc[:, [0,2,3,4,]].values.astype(np.float32) # 0,2,3,4,15,16
output_label = dataset_label.iloc[:, [-1]].values.astype(str)  # Assuming your output is in string format
input_unlabel = dataset_unlabel.iloc[:, [0,2,3,4,]].values.astype(np.float32) #,15,16

# dataset_label = pd.read_csv('/data/fshan/fs/songhua/All_data/Processed9/cycle_label5.csv', header= None, index_col= 0)
# dataset_unlabel = pd.read_csv('/data/fshan/fs/songhua/All_data/Processed9/cycle_unlabel5.csv', header= None, index_col= 0)
# input_label = dataset_label.iloc[:, :-4].dropna(axis=1).values.astype(np.float32)
# output_label = dataset_label.iloc[:, [-1]].values.astype(str)  # Assuming your output is in string format
# input_unlabel = dataset_unlabel.iloc[:, :-4].dropna(axis=1).values.astype(np.float32)

# min-max normalisation
scaler = MinMaxScaler()
input_mm = scaler.fit_transform(input_label)
input_unmm = scaler.transform(input_unlabel)

# Encode class labels into numerical values
label_encoder = LabelEncoder()
output_label_encoded = label_encoder.fit_transform(output_label.ravel())

# Split labeled data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(input_mm, output_label_encoded, test_size=0.2, random_state=42)

# Balance the classes using SMOTE
smote = SMOTE(random_state=42, ) #sampling_strategy={0:100,3:100},
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

model_svc = SVC(probability=True, random_state=42) #kernel='rbf', C=1.0, gamma='scale',
model_rf = RandomForestClassifier(random_state=42)
model_gboost = GradientBoostingClassifier(random_state=42)

# hyperparameters
param_svc = {
    'C': [0.1, 1, 5, 10, 50, 100],
    'kernel': ['poly','linear','rbf' ],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    # 'degree': [2, 3, 4],  # Only relevant for 'poly' kernel
}
param_rf = {
    'n_estimators': [20, 50, 100, 150, 200],
    'max_depth': [None, 5, 7, 10, 20, 50],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 5],
}
param_gboost = {
    'n_estimators': [20, 50, 100, 150, 200],
    'learning_rate':[0.01, 0.05, 0.1, 0.5 ],
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [1, 2, 5],
    'subsample': [0.8, 0.9]
}

random_search = RandomizedSearchCV(model_gboost, param_gboost, n_iter=60, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_resampled, Y_train_resampled.ravel())
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

train_pred = best_model.predict(X_train)
f1_scores_per_class_train = f1_score(Y_train, train_pred, average=None)
print("F1 Score per class in training:", f1_scores_per_class_train)

test_pred = best_model.predict(X_test)
f1_scores_per_class_test = f1_score(Y_test, test_pred, average=None)
print("F1 Score per class in test:", f1_scores_per_class_test)
f1_micro = f1_score(Y_test, test_pred, average='micro')
print("Micro-averaged F1 score:", f1_micro)
f1_macro = f1_score(Y_test, test_pred, average='macro')
print("Macro-averaged F1 score:", f1_macro)

# Calculate confusion matrix
confusion = confusion_matrix(test_pred, Y_test)
class_names = [str(x) for x in label_encoder.classes_]

# Convert y-tick values to Roman numerals
ticks = range(2, 6, 1)
tick_labels = [roman.toRoman(x) for x in ticks]

# Print confusion matrix
plt.figure(figsize=(6.5, 6),dpi=150)
# plt.title('(a) SMOTE-SVM Model',fontsize = 22)
# plt.title('(b) SMOTE-RF Model',fontsize = 22)
# plt.title('(c) SMOTE-GBDT Model',fontsize = 22)
ax = sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True, vmin=0, vmax=30,
                 xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 22})
cbar = ax.collections[0].colorbar  # Get the color bar
cbar.set_ticks([0,10,20,30])
cbar.set_ticklabels(['0','10','20','30'])
cbar.ax.tick_params(length =0, labelsize=16)

plt.xlabel('Actual', fontsize=20)
plt.ylabel('Predicted', fontsize=20)

# Move the ticks to the center of the cells
ax.set_xticks([x + 0.5 for x in range(len(class_names))], minor=False)
ax.set_yticks([y + 0.5 for y in range(len(class_names))], minor=False)

# Set tick labels to Roman numerals
ax.set_xticklabels(tick_labels, minor=False, fontsize=18)
ax.set_yticklabels(tick_labels, minor=False, fontsize=18)
plt.tight_layout()
plt.show()
