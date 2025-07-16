# Repository for "Rock Mass Classification from TBM Operations: Addressing Label Scarcity Using Semi-Supervised Learning"

This repository contains the Python scripts implementing the machine learning algorithms presented in **Section 5** of our **Rock Mass Classification from TBM Operations: Addressing Label Scarcity Using Semi-Supervised Learning**.

## Structure

The following scripts correspond to specific sections of the paper:

### Section 5.2 — Supervised Learning Model

* **`Classifier_basic.py`**
  Implements the baseline supervised learning classifiers used for performance evaluation. Includes preprocessing, training, and testing using limited labeled data.

### Section 5.3 — Handling Data Imbalance

* **`SMOTE.py`**
  Applies the **Synthetic Minority Oversampling Technique (SMOTE)** to mitigate class imbalance issues for supervised classifiers.

### Section 5.4 — Semi-Supervised Learning

* **`self-tra.py`**
  Implements a **self-training semi-supervised learning** framework that leverages a small labeled dataset and a large pool of unlabeled data.

## Requirements

* Python ≥ 3.8
* scikit-learn
* imbalanced-learn
* numpy
* pandas

