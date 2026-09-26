# Dataset

**Name:** Human Activity Recognition Using Smartphones (HAR)
**Source:** UCI Machine Learning Repository
**Link:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
**License:** CC BY 4.0

## Description

Recordings from 30 volunteers (age 19–48) performing six activities of daily
living — WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING,
LAYING — while wearing a waist-mounted Samsung Galaxy S II smartphone. 3-axial
linear acceleration and 3-axial angular velocity were captured at 50Hz and
processed into 561 time- and frequency-domain features per sample.

- **Total samples:** 10,299
- **Train / test split:** 7,352 / 2,947 (~70% / 30%), split by subject
- **Features:** 561, described in `features.txt`
- **Classes:** 6 activities, described in `activity_labels.txt`
- **Files provided by UCI:** `features.txt`, `activity_labels.txt`,
  `train/X_train.txt`, `train/y_train.txt`, `train/subject_train.txt`,
  `test/X_test.txt`, `test/y_test.txt`, `test/subject_test.txt`

## Obtaining the data

This repository does not include the raw dataset. Download it directly from
the UCI Machine Learning Repository link above (or via `ucimlrepo`/Kaggle
mirrors) and place it locally, e.g.:

```
data/
└── UCI HAR Dataset/
    ├── features.txt
    ├── activity_labels.txt
    ├── train/
    └── test/
```

Both notebooks read this path from the `HAR_DATA_DIR` environment variable
(defaulting to `data/UCI HAR Dataset`), so you don't need to edit the code —
just place the data there or point the variable at wherever you downloaded it.

## Citation

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2012). Human
Activity Recognition Using Smartphones. UCI Machine Learning Repository.
https://doi.org/10.24432/C54S4K
