# Project: Survival Prediction Using Logistic Regression

## Files
- survival_prediction.ipynb : Jupyter notebook implementing the end-to-end logistic regression pipeline
- Titanic-Dataset.csv                       : Dataset used for survival prediction
Data Source - Kaggle [Titanic Dataset.csv](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

## Objective
Predict whether a passenger survived or not using logistic regression based on passenger attributes (e.g., age, sex, class).

## Dataset Description (`titanic.csv`)
Typical columns used:
- `PassengerId`: Unique ID of each passenger
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`, `Sex`, `Age`
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`, `Fare`, `Cabin`
- `Embarked`: Port of embarkation
- `Survived`: Target (0 = No, 1 = Yes)

##  Key Steps in Notebook
1. Load and inspect the data
2. EDA
3. Handle missing values (e.g., Age, Embarked)
4. Encode categorical variables (`Sex`, `Embarked`)
5. Feature engineering (e.g., extract titles from names, family size)
6. Split into train-test sets
7. Train a logistic regression model using scikit-learn
8. Evaluate using accuracy, confusion matrix, precision, recall, F1-score
9. Predict survival on test set

## Requirements
- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

To install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```
## How to Run
*  Open titanic_logistic_regression.ipynb in Jupyter Notebook or VS Code

* Ensure titanic.csv is in the same folder

* Run all cells sequentially to see preprocessing, model training, and predictions

##  Output
* Confusion matrix and classification report

* Survival predictions for test set


