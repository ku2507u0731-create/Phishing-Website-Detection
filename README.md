# Phishing Website URL Detection Using Machine Learning

A student machine-learning project that classifies website records as **phishing** or **legitimate** using four supervised learning algorithms.

## Project contents

```text
Phishing-Website-Detection/
├── dataset/
│   └── dataset_phishing.csv
├── IML_PROJECT.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset details

- Records: **11,430**
- Columns: **89**
- Input features: **87 numeric features**
- Excluded from training: `url` (raw URL text)
- Target column: `status`
- Classes: 5,715 legitimate and 5,715 phishing
- Target mapping used in the notebook: `phishing = 1`, `legitimate = 0`

## Models used

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN, 5 neighbors)
4. Random Forest (100 trees)

Logistic Regression and KNN use StandardScaler. The notebook uses a stratified 80/20 train-test split with random state 42.

## Test-set results

Results from the included notebook run:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9361 | 0.9392 | 0.9326 | 0.9359 |
| Decision Tree | 0.9339 | 0.9291 | 0.9396 | 0.9343 |
| KNN | 0.9396 | 0.9539 | 0.9239 | 0.9387 |
| Random Forest | 0.9593 | 0.9549 | 0.9641 | 0.9595 |

Random Forest achieved the highest F1-score in this run. Results can vary with library versions or changes to the data.

## Run the project locally

### 1. Install Python

Install Python 3.10 or newer.

### 2. Download the repository

Clone it using:

```bash
git clone https://github.com/YOUR-USERNAME/Phishing-Website-Detection.git
cd Phishing-Website-Detection
```

Or use GitHub's **Code → Download ZIP**, then extract the ZIP.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook
```

Open `IML_PROJECT.ipynb` and select **Kernel → Restart & Run All**.

If Jupyter is not installed, install it with:

```bash
python -m pip install notebook
```

## Evaluation

The notebook reports:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- Random Forest feature importance

## Limitations and responsible use

This is an academic demonstration, not a production-grade browser protection service. It uses a single public dataset and one train-test split. High test scores do not guarantee performance on new or unseen websites. Do not enter passwords or sensitive information based on this model's prediction, and do not use its output as the only security check.

## Authors

- Hom Patel
- Kiran Rabari

Karnavati University — CSE-CYBER
