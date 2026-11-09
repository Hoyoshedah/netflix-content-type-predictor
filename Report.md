# Report: Content Type Prediction Model

**Auspify Technologies Internship, Task 2 (Easy)**

## 1. Goal

Build a machine learning model that predicts if a Netflix title is a **Movie** or a **TV Show** from its features.

## 2. The data

- 8,790 rows and 10 columns: show_id, type, title, director, country, date_added, release_year, rating, duration, listed_in
- I removed 3 repeated titles (same title, type and release year), leaving 8,787 rows
- The text "Not Given" was used for missing values (director 2,587 and country 287). I turned it into real blanks
- 6,124 Movies (69.7%) and 2,663 TV Shows (30.3%). The classes are not equal, so accuracy alone can mislead. A model that always says "Movie" scores 69.7%. I call this the **baseline**

## 3. Step 1: Select relevant features

| Column                   | Decision       | Reason                              |
| ------------------------ | -------------- | ----------------------------------- |
| show_id, title           | Dropped        | Names and IDs, nothing to learn     |
| type                     | Target         | What we predict                     |
| duration                 | Dropped        | Leaks the answer                    |
| listed_in                | Kept, cleaned  | Removed "tv", "movies", "shows"     |
| country                  | Kept           | First country, top 10 plus "Other"  |
| director                 | Kept as yes/no | Too many names, so "has a director" |
| rating                   | Kept           |                                     |
| release_year, date_added | Kept           | Year and month                      |

### The data leakage problem

`duration` says "min" for every Movie and "Season" or "Seasons" for every TV Show (see `screenshots/02_leakage_demo.png`). A model using it scores about 100%, but that is cheating, because the answer is hidden inside the input. This is called **data leakage**.

I ran three experiments to show the difference:

1. **Leaky:** all features plus duration
2. **Honest:** no duration, cleaned genres
3. **Strict:** no duration and no genres (genre names like "Reality" can still hint at the type)

## 4. Step 2: Encode categorical variables

A model only understands numbers, so I converted the words:

- **One-hot encoding** for rating and country: one 0/1 column per value. I did not use 1, 2, 3, because that would make the model think one rating is "bigger" than another
- **Genres:** cleaned and turned into 0/1 columns with CountVectorizer
- **Director:** turned into 1 (has a director) or 0 (blank)
- **date_added:** split into year and month
- The target: Movie = 0, TV Show = 1

The final input table had 61 columns (29 in the strict version).

## 5. Step 3: Train classification models

I split the data into 80% training (7,029 titles) and 20% test (1,758 titles), with `stratify` so both parts keep the 70/30 mix and `random_state=42` so the split is repeatable. All three experiments use the same test titles, so the comparison is fair.

Four models:

- **Logistic Regression:** a score from the features, and "TV Show" if the score is high enough
- **Decision Tree:** a chain of yes/no questions
- **Random Forest:** 200 trees that vote
- **KNN:** copies the majority of the 5 most similar training titles

## 6. Step 4: Evaluate performance

Accuracy on the test set:

| Model               | Leaky | Honest | Strict |
| ------------------- | ----- | ------ | ------ |
| Logistic Regression | 1.000 | 0.989  | 0.948  |
| Decision Tree       | 1.000 | 0.978  | 0.918  |
| Random Forest       | 1.000 | 0.987  | 0.940  |
| KNN                 | 0.994 | 0.971  | 0.939  |

Baseline: 0.697.

For the best strict model (Logistic Regression): precision 0.933, recall 0.893, F1 0.913.

**Confusion matrix** (`screenshots/05_confusion_matrix.png`):

|              | Predicted Movie | Predicted TV Show |
| ------------ | --------------- | ----------------- |
| Real Movie   | 1,191           | 34                |
| Real TV Show | 57              | 476               |

The model's most common mistake is calling a TV Show a Movie (57 times).

## 7. Step 5: Compare model accuracy

- Logistic Regression was the best in all three experiments, followed by Random Forest. The gaps between models are small (about 1 to 3 points)
- **5-fold cross-validation** gave nearly the same numbers (honest: 0.989, strict: 0.952 for Logistic Regression), with tiny `Std` values (0.002 to 0.006). So the results are stable and not just a lucky split
- The strict model beats the baseline by about 25 points, which is real learning

## 8. What the model looks at

The Random Forest feature chart (`screenshots/06_feature_importance.png`) shows `has_director` is the most important feature by far (about 0.69), then month_added and release_year. Pakistan, India and South Korea also appear, because those countries have many TV dramas on Netflix.

## 9. Limitations

- `has_director` is strong because many TV Shows in this dataset have no director listed. This is a quirk of the data and may not be true for other data
- No description or cast columns exist, which could make predictions better
- Some genre names still hint at the type, which is why the honest result is so high
- With 1,758 test titles, small differences between models may be noise

## 10. What I learned

- Data leakage: I found that the duration column gave the answer away. Every movie says "min" and every TV show says "Season", so a model could get 100% without really learning. I removed it and tested again to get a fair result.
- Baseline:About 70% of the titles are movies, so a model that always says "Movie" is already 70% right. This taught me that accuracy alone can fool you, and that a model must beat the baseline.
- Encoding: Models only understand numbers. I used one-hot encoding to turn ratings and countries into 0 and 1 columns. I did not use 1, 2, 3, because that would make the model think one rating is bigger than another.
- Train and test split: I kept 20% of the titles hidden while the model learned, then used them like an exam. This shows how the model does on titles it has never seen.
- Comparing models: I trained four models. They scored close to each other, and Logistic Regression was the best by a small amount. Cross-validation gave nearly the same numbers, so I trust the result.
- A surprise: The most important feature was whether a title has a director listed. In this dataset many TV shows have no director, so this is a quirk of the data and not a rule about TV.

## 11. Files

`notebooks/classifier.ipynb` has all the code. `screenshots/` has 6 charts and tables. The environment is listed in `requirements.txt`.
