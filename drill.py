"""
Module 5 Week B — Core Skills Drill: Tree-Based Model Basics

Complete the three functions below.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/telecom_churn.csv")

features = ["tenure","monthly_charges","total_charges","num_support_calls","senior_citizen","has_partner","has_dependents",
"contract_months"
]

X = df[features]
y = df["churned"]

print("NaNs in features:", X[features].isnull().sum())
print("Data types:\n", X[features].dtypes)
X['total_charges'] = pd.to_numeric(X['total_charges'], errors='coerce').fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
def train_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train,y_train)
    return model



def get_feature_importances(model, feature_names):
   importances = model.feature_importances_
   pairs = zip (feature_names,importances)
   sorted_pairs = sorted (pairs,
    key = lambda x : x[1],
    reverse= True
)    
   return dict(sorted_pairs)

def train_balanced_forest(X_train, y_train, X_test, y_test,
                          n_estimators=100, random_state=42):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        min_samples_split=5,
        random_state=random_state
    )
    model.fit(X_train,y_train)
    y_pred =model.predict(X_test)
    print("RF predictions unique:", np.unique(y_pred, return_counts=True))
    print("Test set distribution:", np.bincount(y_test))

    return{
        "precision":precision_score(y_test,y_pred,average='binary'),
        "recall":recall_score(y_test,y_pred,average ='binary'),
        "f1":f1_score(y_test,y_pred,average='binary')
    }
    
if __name__ == "__main__":
    df = pd.read_csv("data/telecom_churn.csv")
    features = ["tenure", "monthly_charges", "total_charges",
                "num_support_calls", "senior_citizen", "has_partner",
                "has_dependents", "contract_months"]
    X = df[features]
    y = df["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Task 1
    tree = train_decision_tree(X_train, y_train)
    if tree:
        print(f"Decision tree trained, depth={tree.get_depth()}")

    # Task 2
    if tree:
        importances = get_feature_importances(tree, features)
        if importances:
            print(f"Top features: {list(importances.items())[:3]}")

    # Task 3
    metrics = train_balanced_forest(X_train, y_train, X_test, y_test)
    if metrics:
        print(f"Balanced RF: {metrics}")
