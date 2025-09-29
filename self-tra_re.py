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
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, ParameterSampler, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
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

# Concatenate labeled and unlabeled data
X_combined = np.vstack((X_train, X_unlabel))
Y_combined = np.hstack((y_train, np.full(X_unlabel.shape[0], -1)))  # -1 represents unlabeled samples

param_svc = {
    'estimator__C': [0.1, 1, 5, 10, 50, 100, ], #
    'estimator__kernel': ['poly','linear','rbf'],#'poly','linear','rbf'
    'estimator__gamma': [0.001, 0.01, 0.1, 1, 10], #0.001, 0.01, 0.1, 1, 10
    'threshold': [0.8,0.85,0.9], # 0.8,0.85,0.9
    'max_iter': [1,3,5,7] #1,3,5,7
}
param_rf = {
    'estimator__n_estimators': [20, 50, 100, 150, 200],#
    'estimator__max_depth': [None, 5, 7, 10, 20, 50],#
    'estimator__min_samples_split': [2, 3, 5],#
    'estimator__min_samples_leaf': [1, 2, 5],#
    'threshold': [0.8,0.85,0.9], #0.75-5, 0.8-7, 0.85-10
    'max_iter': [1,3,5,7]#
}
param_gboost = {
    'estimator__n_estimators': [20, 50, 100, 150, 200],#
    'estimator__learning_rate':[0.01, 0.05, 0.1, 0.5],#
    'estimator__max_depth': [3, 5, 7, 10],#
    'estimator__min_samples_leaf': [1, 2, 5],#
    'estimator__subsample': [0.8, 0.9],#
    'threshold': [0.8,0.85,0.9], # 0.8-4, 0.85-5, 0.9-5
    'max_iter': [1,3,5,7]#
}

model_svc = SVC(probability=True, random_state=42)
model_rf = RandomForestClassifier(random_state=42)
model_gboost = GradientBoostingClassifier(random_state=42)

model_selftra = SelfTrainingClassifier(estimator=model_gboost, verbose=False)
random_search = RandomizedSearchCV(
    model_selftra, param_gboost, n_iter=60, cv=5, scoring='accuracy', verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_combined, Y_combined)

best_model = random_search.best_estimator_
print("Best parameters:", random_search.best_params_)

train_pred = best_model.predict(X_train)
f1_per_class_train = f1_score(y_train, train_pred, average=None)
print("F1 per class in training:", f1_per_class_train)

Y_pred_test = best_model.predict(X_test)
Y_proba_test = best_model.predict_proba(X_test)
f1_per_class_test = f1_score(y_test, Y_pred_test, average=None)
print("F1 per class in test:", f1_per_class_test)
f1_micro = f1_score(y_test, Y_pred_test, average='micro')
print("Micro-averaged F1:", f1_micro)
f1_macro = f1_score(y_test, Y_pred_test, average='macro')
print("Macro-averaged F1:", f1_macro)

# ---- Top-2 Accuracy
def top_k_accuracy(y_true, y_proba, k=2):
    topk = np.argsort(-y_proba, axis=1)[:, :k]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])

top2 = top_k_accuracy(y_test, Y_proba_test, k=2)
print("Top-2 Accuracy (test):", top2)

# # Calculate confusion matrix
# confusion = confusion_matrix(Y_test, Y_pred_test)
# class_names = [str(x) for x in label_encoder.classes_]
#
# # Convert y-tick values to Roman numerals
# ticks = range(2, 6, 1)
# tick_labels = [roman.toRoman(x) for x in ticks]
#
# # Print confusion matrix
# plt.figure(figsize=(6.5, 6),dpi=150)
# # plt.title('(a) SSL-SVM Model',fontsize = 22)
# # plt.title('(b) SSL-RF Model',fontsize = 22)
# # plt.title('(c) SSL-GBDT Model',fontsize = 22)
# ax = sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True, vmin=0, vmax=30,
#                  xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 22})
# cbar = ax.collections[0].colorbar  # Get the color bar
# cbar.set_ticks([0,10,20,30])
# cbar.set_ticklabels(['0','10','20','30'])
# cbar.ax.tick_params(length =0, labelsize=16)
#
# plt.xlabel('Predicted', fontsize=20)
# plt.ylabel('Actual', fontsize=20)
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

