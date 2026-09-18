import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

SEED = 42

DROP_COLS = ['PassengerId', 'Cabin', 'Name', 'Ticket']

train_df = train_df.drop(DROP_COLS, axis=1)

X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=SEED)

categorical_columns = ['Sex', 'Pclass', 'Embarked']
numerical_columns = ['Age', 'Fare']

categorical_transformer = Pipeline(steps=[
      ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
      ('imputer', SimpleImputer(strategy='median'))
])

scaler = Pipeline(steps=[
      ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
      ('categorical_preprocessor', categorical_transformer, categorical_columns),
      ('numerical_columns', numerical_transformer, numerical_columns)
], remainder='passthrough')

model_pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', XGBClassifier(
            learning_rate=0.01,
            max_depth=5,
            n_estimators=500,
            eval_metric='auc',
            random_state=SEED,
            tree_method='hist',
            scale_pos_weight=3.8))
])

skf = StratifiedKFold(n_splits=10, random_state=SEED, shuffle=True)

accuracy_scores = []
roc_auc_scores = []

for train_index, test_index in skf.split(X, y):
      X_train = X.iloc[train_index, :]
      X_test = X.iloc[test_index, :]
      y_train = y[train_index]
      y_test = y[test_index]

      model_pipeline.fit(X_train, y_train)

      preds = model_pipeline.predict(X_test)
      probs = model_pipeline.predict_proba(X_test)[:, 1]

      roc_auc = roc_auc_score(y_test, probs)
      roc_auc_scores.append(roc_auc)

      accuracy = accuracy_score(y_test, preds)
      accuracy_scores.append(accuracy)

print(f'Average Accuracy: {np.round(sum(accuracy_scores) / len(accuracy_scores), 4)}')
print(f'Average ROC-AUC Score: {np.round(sum(roc_auc_scores) / len(roc_auc_scores), 4)}')

row_ids = test_df['PassengerId']

test_df = test_df.drop(DROP_COLS, axis=1)
test_preds = model_pipeline.predict(test_df)

submission = pd.DataFrame({
      'PassengerId': row_ids,
      'Survived': test_preds
})

submission.to_csv('./script/submission.csv', index=False)