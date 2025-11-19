# classification_pipeline.py

# =====================================================
# STEP 0: Imports 
# =====================================================
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =====================================================
# STEP 1: Load dataset
# =====================================================
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('clustered_cars_data.csv')
print(f" Loaded clustered dataset: {df.shape}")
print(f"Unique clusters: {sorted(df['Final_Cluster'].unique())}\n")

identifier_cols = [col for col in ['company_name', 'car_name'] if col in df.columns]

numeric_features = [
    'engine_displacement_in_cc',
    'battery_energy_capacity_in_kwh',
    'horsepower_in_hp',
    'torque_in_nm',
    'performance_0_to_100_km_per_h',
    'total_speed_in_km_per_h',
    'seats_parsed',
    'cars_price_amount',
    'value_for_money'
]

categorical_features = [
    'fuel_types_normalized',
    'company_segment'
]

target = 'Final_Cluster'

X = df[numeric_features + categorical_features].copy()
y = df[target].astype(int)

# =====================================================
# STEP 2: Handle missing values
# =====================================================
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X[categorical_features] = X[categorical_features].fillna("Unknown")

# Label mapping
y_unique = sorted(y.unique())
y_mapping = {old: i for i, old in enumerate(y_unique)}
reverse_mapping = {v: k for k, v in y_mapping.items()}
y = y.map(y_mapping)

print("Cluster label mapping:", y_mapping)

# =====================================================
# STEP 3: Preprocessor
# =====================================================
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# =====================================================
# STEP 4: Define classifiers
# =====================================================
classifiers = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_split=10,
        min_samples_leaf=5, max_features='sqrt', class_weight='balanced',
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=250, learning_rate=0.07, max_depth=3,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0,
        reg_lambda=5.0, eval_metric='mlogloss', use_label_encoder=False,
        random_state=42
    )
}

# =====================================================
# STEP 5: 5-Fold Cross-Validation 
# =====================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}  # metrics per model
pipes = {}    # trained models

for name, clf in classifiers.items():
    pipe = Pipeline([('preprocessor', preprocessor),
                     ('classifier', clf)])
    
    # Store fold metrics
    acc_list, f1_list, prec_list, rec_list = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)

        acc_list.append(accuracy_score(y_val, y_pred))
        f1_list.append(f1_score(y_val, y_pred, average='macro'))
        prec_list.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        rec_list.append(recall_score(y_val, y_pred, average='macro', zero_division=0))

    # Average metrics over folds
    results[name] = {
        "Accuracy": np.mean(acc_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1": np.mean(f1_list)
    }

    print(f"\n{name} 5-Fold CV Results (Training Data):")
    print(f"Accuracy: {results[name]['Accuracy']:.3f}")
    print(f"Precision: {results[name]['Precision']:.3f}")
    print(f"Recall: {results[name]['Recall']:.3f}")
    print(f"F1 Score: {results[name]['F1']:.3f}")

    # Train final model on full dataset for deployment
    pipe.fit(X, y)
    pipes[name] = pipe

# =====================================================
# STEP 6: Select Best Model based on CV F1
# =====================================================
best_name = max(results, key=lambda m: results[m]["F1"])
best_pipe = pipes[best_name]

joblib.dump(best_pipe, "models/best_classification_model.joblib")
joblib.dump(reverse_mapping, "models/reverse_mapping.joblib")

print(f"\nBest Model based on 5-Fold CV: {best_name} saved to 'models/'")

# =====================================================
# STEP 7: Plot Comparison of Metrics
# =====================================================
metrics = ["Accuracy", "Precision", "Recall", "F1"]
plot_df = pd.DataFrame(results).T

plt.figure(figsize=(10,6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#ffdd00"]  # metrics colors
plot_df[metrics].plot(kind='bar', color=colors)
plt.title("Model Comparison (5-Fold CV on Training Data)")
plt.ylabel("Score")
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(title="Metric", loc="lower right")
plt.tight_layout()
plt.savefig("plots/model_comparison_metrics.png")
plt.show()


# =====================================================
# STEP 8: Recommendation Function
# =====================================================
def predict_customer_segment(customer_dict, top_n=5):
    model = joblib.load('models/best_classification_model.joblib')
    reverse_mapping = joblib.load('models/reverse_mapping.joblib')

    input_df = pd.DataFrame([customer_dict])
    price_input = customer_dict.get('cars_price_amount')

    if isinstance(price_input, dict):
        min_price = price_input.get('min', X['cars_price_amount'].min())
        max_price = price_input.get('max', X['cars_price_amount'].max())
        input_df['cars_price_amount'] = (min_price + max_price) / 2
    else:
        input_df['cars_price_amount'] = price_input

    for f in numeric_features:
        if f not in input_df:
            input_df[f] = X[f].median()
    for f in categorical_features:
        if f not in input_df:
            input_df[f] = 'Unknown'

    input_df = input_df[numeric_features + categorical_features]
    cluster_pred = model.predict(input_df)[0]
    cluster_original = reverse_mapping.get(cluster_pred, cluster_pred)
    print(f"\ Predicted Cluster: {cluster_original}")

    recs = df[df['Final_Cluster'] == cluster_original].copy()
    recs.loc[:, numeric_features] = recs[numeric_features].fillna(df[numeric_features].median())

    if isinstance(price_input, dict):
        recs = recs[(recs['cars_price_amount'] >= min_price) &
                    (recs['cars_price_amount'] <= max_price)]
    else:
        recs['price_diff'] = abs(recs['cars_price_amount'] - price_input)
        recs = recs.sort_values('price_diff')

    recs = recs.head(top_n)
    display_cols = identifier_cols + numeric_features + categorical_features + ['value_for_money']

    print("\nTop Recommended Cars:")
    print(recs[display_cols].to_string(index=False))

    return cluster_original, recs, display_cols

# =====================================================
# STEP 9: Example Usage
# =====================================================
if __name__ == "__main__":
    example_customer = {
        'cars_price_amount': {'min': 25000, 'max': 40000},
        'seats_parsed': 5,
        'fuel_types_normalized': 'Petrol',
        'company_segment': 'Mass_Market_Mainstream'
    }

    segment, recs, cols = predict_customer_segment(example_customer, top_n=5)
    recs[cols].to_csv("top_recommendations.csv", index=False)
    print("\n Recommendations exported to top_recommendations.csv")
