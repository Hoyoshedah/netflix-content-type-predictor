# Netflix Content Type Prediction Model

**Auspify Technologies Internship, Task 2**

A machine learning project that predicts whether a Netflix title is a **Movie** or a **TV Show**, using the information in the dataset.

## What this project does

1. Selects the useful columns from the dataset
2. Turns words into numbers (encoding)
3. Trains 4 classification models
4. Tests how well each model predicts
5. Compares the models

## Important finding: data leakage

The `duration` column gives the answer away. Movies say "90 min" and TV Shows say "1 Season". A model can get 100% by reading this one column, but it learns nothing useful.

So I ran **three experiments**:

| Experiment | What the model can see                           |
| ---------- | ------------------------------------------------ |
| 1. Leaky   | Everything, including duration                   |
| 2. Honest  | No duration, cleaned genres                      |
| 3. Strict  | No duration and no genres (the most trustworthy) |

## Dataset

- File: `data/Dataset.csv`
- 8,790 rows, 10 columns. After removing 3 repeated titles: 8,787 rows
- 6,124 Movies and 2,663 TV Shows (about 70% / 30%)

## Results (accuracy on the 20% test set)

| Model               | 1. Leaky | 2. Honest | 3. Strict |
| ------------------- | -------- | --------- | --------- |
| Logistic Regression | 1.000    | 0.989     | 0.948     |
| Decision Tree       | 1.000    | 0.978     | 0.918     |
| Random Forest       | 1.000    | 0.987     | 0.940     |
| KNN                 | 0.994    | 0.971     | 0.939     |

- **Baseline** (always guessing "Movie"): 0.697
- **Best trustworthy model:** Logistic Regression, strict experiment, 94.8% accuracy (95.2% in 5-fold cross-validation)
- The differences between the models are small, about 1 to 3 points

## Folder structure

```
Content Type Predictor/
  data/Dataset.csv
  notebooks/classifier.ipynb
  screenshots/        (6 charts and tables)
  README.md
  Report.md
  requirements.txt
  .gitignore
```

## How to run it (Windows PowerShell)

```powershell
cd "D:\Content Type Predictor"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks\classifier.ipynb
```

Then click **Run All**.

## Tools

Python 3.10, pandas, numpy, scikit-learn, matplotlib, Jupyter, VS Code, Git/GitHub

## Limitations

- `has_director` is the strongest feature, but that is a quirk of this dataset (many TV Shows have no director listed). It may not hold for other data.
- The dataset has no description or cast column, which could help a lot.
- The test set has 1,758 titles, so small differences between models could be noise.

## Links

GitHub repo: https://github.com/Hoyoshedah/netflix-content-type-predictor
