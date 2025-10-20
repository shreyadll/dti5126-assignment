# classification_pipeline_final.py
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

# ================================================
# STEP 1: Load clustered dataset
# ================================================
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('clustered_cars_data.csv')
print(f" Loaded clustered dataset: {df.shape}")
print(f"Unique clusters: {sorted(df['Final_Cluster'].unique())}\n")

# Identify non-feature columns
identifier_cols = [col for col in ['company_name', 'car_name'] if col in df.columns]

# ================================================
# STEP 2: Define features and target
# ================================================
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

X = df[numeric_features + categorical_features]
y = df[target].astype(int)

# ================================================
# STEP 3: Handle missing values and map labels
# ================================================
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X[categorical_features] = X[categorical_features].fillna('Unknown')

y_unique = sorted(y.unique())
y_mapping = {old: i for i, old in enumerate(y_unique)}
reverse_mapping = {v: k for k, v in y_mapping.items()}
y = y.map(y_mapping)

print("Cluster label mapping:", y_mapping)

# ================================================
# STEP 4: Preprocessing setup
# ================================================
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ================================================
# STEP 5: Define classifiers
# ================================================
classifiers = {
    "RandomForest": RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight='balanced'
    ),
    "XGBoost": XGBClassifier(
        random_state=42,
        n_estimators=350,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
}

results = []
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================================
# STEP 6: Train, Evaluate, Plot Confusion Matrices
# ================================================
for name, clf in classifiers.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n {name} Results:")
    print(f"Cross-val F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test F1 Macro: {f1:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    labels = [f"Cluster {lbl}" for lbl in sorted(y.unique())]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{name} Confusion Matrix (Test Set)")
    plt.tight_layout()

    # Save confusion matrix image
    plot_path = f"plots/{name}_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.show()

    print(f" Confusion matrix saved to {plot_path}\n")

    results.append((name, scores.mean(), acc, f1, pipe))

# ================================================
# STEP 7: Select Best Model and Save
# ================================================
best_model = max(results, key=lambda x: x[2])
best_name, best_cv, best_acc, best_f1, best_pipe = best_model

print(f"\n Best Model: {best_name}")
print(f"   CV F1: {best_cv:.3f} | Accuracy: {best_acc:.3f} | F1: {best_f1:.3f}")

joblib.dump(best_pipe, 'models/best_classification_model.joblib')
joblib.dump(reverse_mapping, 'models/reverse_mapping.joblib')
print(" Model and mapping saved to 'models/'")

# ================================================
# STEP 8: Cross-Validated Confusion Matrix (Best Model)
# ================================================
print("\n Generating Cross-Validated Confusion Matrix for Best Model...")
y_pred_cv = cross_val_predict(best_pipe, X, y, cv=5)
cm_cv = confusion_matrix(y, y_pred_cv)
labels = [f"Cluster {lbl}" for lbl in sorted(y.unique())]

plt.figure(figsize=(6, 5))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"{best_name} Cross-Validated Confusion Matrix")
plt.tight_layout()
cv_plot_path = f"plots/{best_name}_crossval_confusion_matrix.png"
plt.savefig(cv_plot_path)
plt.show()
print(f" Cross-validated confusion matrix saved to {cv_plot_path}\n")

# ================================================
# STEP 9: Recommendation Function
# ================================================
def predict_customer_segment(customer_dict, top_n=5):
    """
    Predict customer cluster and recommend top-N similar cars.
    """
    model = joblib.load('models/best_classification_model.joblib')
    reverse_mapping = joblib.load('models/reverse_mapping.joblib')

    input_df = pd.DataFrame([customer_dict])

    # Handle price as dict
    price_input = customer_dict.get('cars_price_amount')
    if isinstance(price_input, dict):
        min_price = price_input.get('min', X['cars_price_amount'].min())
        max_price = price_input.get('max', X['cars_price_amount'].max())
        input_df['cars_price_amount'] = (min_price + max_price) / 2
    else:
        input_df['cars_price_amount'] = price_input

    # Fill missing features
    for f in numeric_features:
        if f not in input_df:
            input_df[f] = X[f].median()
    for f in categorical_features:
        if f not in input_df:
            input_df[f] = 'Unknown'

    input_df = input_df[numeric_features + categorical_features]

    # Predict segment
    cluster_pred = model.predict(input_df)[0]
    cluster_original = reverse_mapping.get(cluster_pred, cluster_pred)
    print(f"\ Predicted Cluster: {cluster_original}")

    # Filter recommendations
    recs = df[df['Final_Cluster'] == cluster_original].copy()
    recs[numeric_features] = recs[numeric_features].fillna(df[numeric_features].median())

    if isinstance(price_input, dict):
        recs = recs[
            (recs['cars_price_amount'] >= min_price) &
            (recs['cars_price_amount'] <= max_price)
        ]
    else:
        recs['price_diff'] = abs(recs['cars_price_amount'] - price_input)
        recs = recs.sort_values('price_diff')

    recs = recs.head(top_n)

    # Display columns
    display_cols = identifier_cols + numeric_features + categorical_features + ['value_for_money']

    print("\nTop Recommended Cars:")
    print(recs[display_cols].to_string(index=False))

    return cluster_original, recs, display_cols

# ================================================
# STEP 10: Example usage
# ================================================
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
