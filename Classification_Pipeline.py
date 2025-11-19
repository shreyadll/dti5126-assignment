# classification_pipeline.py

import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =====================================================
# STEP 1: Load clustered dataset
# =====================================================
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('clustered_cars_data.csv')
print(f" Loaded clustered dataset: {df.shape}")
print(f"Unique clusters: {sorted(df['Final_Cluster'].unique())}\n")

identifier_cols = [col for col in ['company_name', 'car_name'] if col in df.columns]

# =====================================================
# STEP 2: Define features and target
# =====================================================
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
# STEP 3: Handle missing values + mapping
# =====================================================
X.loc[:, numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X.loc[:, categorical_features] = X[categorical_features].fillna("Unknown")

# Label mapping
y_unique = sorted(y.unique())
y_mapping = {old: i for i, old in enumerate(y_unique)}
reverse_mapping = {v: k for k, v in y_mapping.items()}
y = y.map(y_mapping)

print("Cluster label mapping:", y_mapping)

# =====================================================
# STEP 4: Preprocessor
# =====================================================
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(
    steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# =====================================================
# STEP 5: Define classifiers 
# =====================================================
classifiers = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    ),

    # XGBoost - tuned to reduce overfitting
    "XGBoost": XGBClassifier(
        n_estimators=250,       # lower tree count
        learning_rate=0.07,     # slightly lower
        max_depth=3,            # shallower trees
        subsample=0.7,          # random row sampling
        colsample_bytree=0.7,   # random feature sampling
        reg_alpha=2.0,          # stronger L1 regularization
        reg_lambda=5.0,         # L2 regularization
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
}

results = []
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# STEP 6: Train Models 
# =====================================================
for name, clf in classifiers.items():

    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', clf)])

    cv_scores = cross_val_score(pipe, X_train, y_train,
                                cv=5, scoring='f1_macro', n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n {name} Results:")
    print(f"Cross-val F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test F1 Macro: {f1:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = [f"Cluster {lbl}" for lbl in sorted(y.unique())]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{name} Confusion Matrix (Test Set)")
    plt.tight_layout()

    plot_path = f"plots/{name}_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.show()

    results.append((name, cv_scores.mean(), acc, f1, pipe))


# =====================================================
# STEP 7: Train Models & Compute Test Metrics 
# =====================================================

results = {}   # store metrics for plotting later
pipes = {}     # store models

for name, clf in classifiers.items():

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    # Train
    pipe.fit(X_train, y_train)

    # Predict
    y_pred = pipe.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Per-class precision/recall → average macro
    from sklearn.metrics import precision_score, recall_score
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Store in dictionary for plotting
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1
    }

    # Save model for later
    pipes[name] = pipe


# =====================================================
# STEP 8: Best Model
# =====================================================
best_name = max(results, key=lambda m: results[m]["F1"])
best_pipe = pipes[best_name]

print(f"\n Best Model: {best_name}")

joblib.dump(best_pipe, "models/best_classification_model.joblib")
joblib.dump(reverse_mapping, "models/reverse_mapping.joblib")
print(" Model and mapping saved to 'models/'")

# =====================================================
# STEP 8: Plot the Result
# =====================================================

metrics = ["Accuracy", "Precision", "Recall", "F1"]

# Prepare data for plot
plot_df = pd.DataFrame(results).T  # models × metrics

plt.figure(figsize=(10, 6))

# Assign a different color to each metric
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#ffdd00"]  # blue, orange, green, red
plot_df[metrics].plot(kind='bar', figsize=(10,6), color=colors)

plt.title("Model Comparison on Test Set")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metric", loc="lower right")
plt.tight_layout()

plt.savefig("plots/model_comparison_metrics.png")
plt.show()

print("\nSaved: plots/model_comparison_metrics.png")



# =====================================================
# STEP 9: Recommendation Function
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
# STEP 10: Example Usage
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
