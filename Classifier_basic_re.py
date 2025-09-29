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

model_svc = SVC(probability=True, random_state=42) #kernel='rbf', C=1.0, gamma='scale',
model_rf = RandomForestClassifier(random_state=42)
model_ann = MLPClassifier(activation='relu',solver='adam',learning_rate='adaptive',batch_size='auto',early_stopping=True,random_state=42)
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
    model_gboost, param_gboost, n_iter=60, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
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

# Confusion matrix (rows = actual, cols = predicted)
cm = confusion_matrix(y_test, y_pred_test, labels=np.arange(n_classes))
class_names = list(label_encoder.classes_)  # safer than hard-coded Roman numerals

plt.figure(figsize=(6.5, 6), dpi=150)
ax = sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=True,
    xticklabels=class_names, yticklabels=class_names
)
cbar = ax.collections[0].colorbar
# Optional: set useful tick marks if your counts are within a known range
cbar.set_ticks([0, 10, 20, 30])
cbar.set_ticklabels(['0', '10', '20', '30'])
cbar.ax.tick_params(length=0, labelsize=12)

plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.tight_layout()
plt.show()

# # # Calculate confusion matrix
# confusion = confusion_matrix(test_pred, Y_test)
# class_names = [str(x) for x in label_encoder.classes_]
#
# # Convert y-tick values to Roman numerals
# ticks = range(2, 6, 1)
# tick_labels = [roman.toRoman(x) for x in ticks]
#
# # Print confusion matrix
# plt.figure(figsize=(6.5, 6),dpi=150)
# # plt.title('(a) SVM Model',fontsize = 22)
# # plt.title('(b) RF Model',fontsize = 22)
# # plt.title('(c) GBDT Model',fontsize = 22)
# ax = sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True, vmin=0, vmax=30,
#                  xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 22})
# cbar = ax.collections[0].colorbar  # Get the color bar
# cbar.set_ticks([0,10,20,30])
# cbar.set_ticklabels(['0','10','20','30'])
# cbar.ax.tick_params(length =0, labelsize=16)
#
# plt.xlabel('Actual', fontsize=20)
# plt.ylabel('Predicted', fontsize=20)
#
# # Move the ticks to the center of the cells
# ax.set_xticks([x + 0.5 for x in range(len(class_names))], minor=False)
# ax.set_yticks([y + 0.5 for y in range(len(class_names))], minor=False)
#
# # Set tick labels to Roman numerals
# ax.set_xticklabels(tick_labels, minor=False, fontsize=18)
# ax.set_yticklabels(tick_labels, minor=False, fontsize=18)
# plt.tight_layout()
#
# plt.show()
