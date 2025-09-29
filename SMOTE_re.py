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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, log_loss
from sklearn.model_selection import train_test_split, RandomizedSearchCV, ParameterSampler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import roman

# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

dataset_label = pd.read_csv(r"D:\Database\Songhua RIver water tunnel\Pro-Data\Processed8\cycle_label2.csv", header= 0, index_col= 0)
dataset_unlabel = pd.read_csv(r"D:\Database\Songhua RIver water tunnel\Pro-Data\Processed8\cycle_unlabel2.csv", header= 0, index_col= 0)
X_label = dataset_label.iloc[:, [0,2,3,4,]].values.astype(np.float32)# 0,2,3,4,15,16
y_label = dataset_label.iloc[:, [-1]].values.astype(str).ravel()  # Assuming your output is in string format
X_unlabel = dataset_unlabel.iloc[:, [0,2,3,4,]].values.astype(np.float32)

# Encode class labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_label)
n_classes = len(label_encoder.classes_)

# Split labeled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_label, y_encoded, test_size=0.2, random_state=42)

# min-max normalisation
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_unlabel = scaler.transform(X_unlabel)

# Balance the classes using SMOTE
smote = SMOTE(random_state=42, ) #sampling_strategy={0:100,3:100},
X_train_os, Y_train_os = smote.fit_resample(X_train, y_train)

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
    'max_depth': [None, 5, 7, 10, 20, 50], # [2, 3, 5, 7, 10]
    'min_samples_split': [2, 3, 5], # 2, 3, 5
    'min_samples_leaf': [1, 2, 5], # 1, 2, 5
}

param_gboost = {
    'n_estimators': [20, 50, 100, 150, 200], #100, 150, 200
    'learning_rate':[0.01, 0.05, 0.1, 0.5 ], #0.01, 0.1, 0.5
    'max_depth': [3, 5, 7, 10], #2, 3,
    'min_samples_leaf': [1, 2, ], #1,
    'subsample': [0.8, 0.9] # 0.9
}

random_search = RandomizedSearchCV(
    model_svc, param_svc, n_iter=60, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_os, Y_train_os)
best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

# Training metrics
y_pred_train = best_model.predict(X_train)
f1_per_class_train = f1_score(y_train, y_pred_train, average=None)
print("F1 per class (train):", f1_per_class_train)

# Test metrics: F1 (micro/macro), Log Loss, Top-2 Accuracy
y_pred_test = best_model.predict(X_test)
y_proba_test = best_model.predict_proba(X_test) # needed for logloss & top-2

f1_per_class_test = f1_score(y_test, y_pred_test, average=None)
f1_micro = f1_score(y_test, y_pred_test, average='micro')
f1_macro = f1_score(y_test, y_pred_test, average='macro')

print("F1 per class (test):", f1_per_class_test)
print("F1 micro (test):", f1_micro)
print("F1 macro (test):", f1_macro)

# ---- Log Loss (use all label indices to be safe even if some classes absent in y_test)
ll = log_loss(y_test, y_proba_test, labels=np.arange(n_classes))
print("Log Loss (test):", ll)

# ---- Top-2 Accuracy
def top_k_accuracy(y_true, y_proba, k=2):
    topk = np.argsort(-y_proba, axis=1)[:, :k]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])

top2 = top_k_accuracy(y_test, y_proba_test, k=2)
print("Top-2 Accuracy (test):", top2)