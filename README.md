# Human Activity Recognition from Smartphone Sensors

Classifying six daily activities (walking, walking upstairs/downstairs, sitting,
standing, laying) from smartphone accelerometer and gyroscope signals, plus an
unsupervised exploration of the same data.

## Overview

This project has two parts, built on the same dataset:

1. **Supervised classification** — four models (KNN, Decision Tree, Random
   Forest, and a Keras ANN) are trained to predict activity type from 561
   pre-computed sensor features, with standardization, PCA, and stratified
   cross-validation.
2. **Unsupervised exploration** — exploratory data analysis and three
   clustering algorithms (KMeans, Agglomerative Clustering, DBSCAN) are used
   to see how well activity classes separate without labels.

## Dataset

UCI "Human Activity Recognition Using Smartphones" dataset: 30 subjects, 6
activity classes, 561 pre-computed sensor features per sample, 10,299 samples
total (7,352 train / 2,947 test). See [`data/README.md`](data/README.md) for
the full description and source link.

## Methodology

### Part 1 — Supervised classification (`notebooks/har_classification.ipynb`)

1. **Data loading** : `features.txt`, `activity_labels.txt`, and the separate
   train/test files (`X_*`, `y_*`, `subject_*`) are loaded and merged into a
   single dataframe of subject, activity label, and 561 measurement columns.
2. **Data cleaning** : no missing values were found in either the training or
   test set; no imputation was required.
3. **Standardization** : features were scaled with `StandardScaler` (zero
   mean, unit variance), fit on the training set only.
4. **Dimensionality reduction (PCA)** : PCA was fit on the standardized
   training data, retaining components covering 95% of total variance. This
   reduced the feature space from 561 to 102 dimensions (an 81.8% reduction).
5. **Cross-validation setup** : stratified 5-fold and stratified 10-fold
   splits were built on the training set only; the test set was kept
   completely separate for final evaluation.
6. **Model training**
   - **KNN** : optimal K was searched over K = 1–25 (odd values) via 5-fold
     cross-validation on accuracy; K=23 was selected. The chosen K was then
     re-evaluated with stratified 5-fold and 10-fold CV.
   - **Decision Tree** : `max_depth`, split criterion (entropy vs. Gini), and
     `max_features` were each swept independently, then combined into a
     single optimized tree, validated with 10-fold CV.
   - **Random Forest** : trained on the PCA-transformed features, validated
     with 10-fold CV, then trained on the full training set and evaluated on
     the held-out test set.
   - **ANN (Keras)** : 3 dense hidden layers (128 → 64 → 32 units, ReLU) with
     Batch Normalization and Dropout (0.3), softmax output over 6 classes,
     Adam optimizer, categorical cross-entropy loss. Evaluated with 10-fold
     CV (10 independently trained models, 90/10 train/validation split per
     fold), then a final model was trained on 100% of the training data and
     evaluated once on the test set.
7. **Evaluation** : accuracy, precision, recall, and F1-score (weighted) were
   computed for every model; confusion matrices and ROC/AUC curves were
   produced for KNN, the Decision Tree, and the ANN.

### Part 2 — EDA and clustering (`notebooks/eda_and_clustering.ipynb`)

1. **Exploratory analysis** : class balance, per-feature distributions
   (histograms, KDE, boxplots), and a static-vs-dynamic activity comparison
   (accelerometer/gyroscope feature counts, magnitude statistics) on the same
   merged train+test dataframe.
2. **Dimensionality reduction for visualization** : PCA (2D) and t-SNE (2D)
   projections of the standardized feature space, colored by activity.
3. **Clustering** : features were standardized, then clustered with:
   - **KMeans** (k=6)
   - **Agglomerative Clustering** (Ward linkage, 6 clusters, plus a
     dendrogram)
   - **DBSCAN** (eps tuned via a k-distance elbow plot; final run used
     `eps=3` on a 3-component PCA projection)
4. **Evaluation** : clusters were compared against the true activity labels
   using Silhouette Score, Davies-Bouldin Score, Adjusted Rand Index (ARI),
   and Adjusted Mutual Information (AMI).

## Results

### Part 1 — Held-out test set (final, most reliable comparison)

| Model                        | Accuracy | Precision | Recall | F1-score |
|-------------------------------|:--------:|:---------:|:------:|:--------:|
| Decision Tree (optimized)      | 77.57%   | 77.77%    | 77.57% | 77.64%   |
| Random Forest (PCA features)   | 88.97%   | 89.45%    | 88.97% | 88.87%   |
| ANN (3 hidden layers)          | 93.62%   | 93.77%    | 93.62% | 93.58%   |

> KNN was not evaluated on the held-out test set in this project — only
> cross-validated training metrics are available (see below).

### Part 1 — Cross-validation on training data (not directly comparable to the table above)

| Model      | CV setup                          | Accuracy | Precision | Recall | F1-score |
|------------|------------------------------------|:--------:|:---------:|:------:|:--------:|
| KNN (K=23) | Stratified 5-fold                  | 96.03%   | 96.05%    | 96.03% | 96.02%   |
| KNN (K=23) | Stratified 10-fold                 | 96.46%   | 96.49%    | 96.46% | 96.45%   |
| ANN        | Stratified 10-fold (mean, val fold) | 97.92%   | 98.11%    | 98.10% | 98.09%   |

### Part 2 — Clustering vs. true activity labels

| Method                     | Silhouette | Davies-Bouldin | ARI    | AMI    |
|----------------------------|:----------:|:--------------:|:------:|:------:|
| KMeans (k=6)                | 0.117      | 2.527           | 0.285  | 0.461  |
| Agglomerative (k=6, Ward)    | 0.106      | 2.472           | 0.426  | 0.587  |
| DBSCAN (eps=3, 3D PCA)       | 0.359      | 2.725           | 0.330  | 0.540  |

> DBSCAN's parameters placed 10,293 of 10,299 points into a single cluster
> (only 6 points flagged as noise). Its labels were produced on a
> 3-component PCA projection while the reported metrics were computed
> against the full standardized 561-feature space.

## Key Findings

- Random Forest and the ANN score highest on the held-out test set; the
  Decision Tree, even after hyperparameter optimization, scores lower
  (~78% vs. ~89–94%).
- PCA reduced the feature space by over 80% while retaining 95% of the
  variance.
- The shuffled stratified CV for KNN reports ~96% accuracy at K=23, while
  the unshuffled CV used during the K-search reported ~88% accuracy for the
  same K. A similar gap appears for the ANN (97.9% CV validation vs. 93.6%
  true test accuracy). Treat the CV-only numbers as upper bounds rather
  than a direct estimate of test performance.
- The Decision Tree's ROC curves show AUC values below 0.5 for several
  classes, inconsistent with its ~78% accuracy — noted as a known issue in
  the notebook.
- None of the three clustering methods cleanly separated all six individual
  activity classes; Agglomerative Clustering had the highest AMI (0.587).

## Project Structure
har-smartphone-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── notebooks/
    ├── har_classification.ipynb
    └── eda_and_clustering.ipynb





## Installation

```bash
git clone <repo-url>
cd har-smartphone-classification
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project

1. Download the dataset (see [`data/README.md`](data/README.md)) and place it
   locally, e.g. under `data/UCI HAR Dataset/`.
2. Both notebooks read the dataset path from the `HAR_DATA_DIR` environment
   variable, defaulting to `data/UCI HAR Dataset`:
```bash
   export HAR_DATA_DIR="/path/to/UCI HAR Dataset"   # Windows: set HAR_DATA_DIR=...
```
   Or just place the dataset at the default path and skip this step.
3. Run either notebook top to bottom with Jupyter or JupyterLab.

## Limitations

- Gap between shuffled cross-validation results and held-out test results
  for KNN and the ANN.
- The Decision Tree's ROC/AUC computation produces AUC values below 0.5 for
  some classes.
- No KNN evaluation exists on the untouched test set.
- Hyperparameter search was limited to a small manual grid per model.
- DBSCAN's cluster assignments and its reported evaluation metrics were
  computed on different feature spaces.
- Models were evaluated on data from the same 30 subjects used for
  training/validation.

## References

- Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2012).
  Human Activity Recognition Using Smartphones. UCI Machine Learning
  Repository. https://doi.org/10.24432/C54S4K
