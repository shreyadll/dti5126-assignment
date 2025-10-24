# ============================================================
# error_analysis.py  (XAI + Error Insights)
# Purpose: Analyze classification errors and interpret model predictions with SHAP
# ============================================================

import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.model_selection import train_test_split

# ============================================================
# STEP 1: Load Data and Model
# ============================================================
print("\n--- ERROR ANALYSIS + EXPLAINABILITY PIPELINE ---\n")

# Load clustered dataset and trained model
df = pd.read_csv("clustered_cars_data.csv")
model = joblib.load("models/best_classification_model.joblib")
reverse_mapping = joblib.load("models/reverse_mapping.joblib")

mapping = {v: k for k, v in reverse_mapping.items()}

# ============================================================
# STEP 2: Define Features and Target
# ============================================================
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
identifier_cols = [col for col in ['company_name', 'car_name'] if col in df.columns]

# Prepare X and y
X = df[numeric_features + categorical_features]
y = df[target].astype(int)

X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X[categorical_features] = X[categorical_features].fillna('Unknown')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# STEP 3: Predictions and Basic Metrics
# ============================================================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"Test Accuracy: {acc:.3f}")
print(f"Macro F1 Score: {f1:.3f}")
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================================
# STEP 4: Confusion Matrix Visualization
# ============================================================
cm = confusion_matrix(y_test, y_pred)
labels = [f"Cluster {lbl}" for lbl in sorted(y.unique())]

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Error Distribution")
plt.tight_layout()
plt.savefig("plots/error_confusion_matrix.png")
plt.show()

print("Confusion matrix saved to 'plots/error_confusion_matrix.png'\n")

# ============================================================
# STEP 5: Misclassification Analysis
# ============================================================
misclassified_mask = y_test != y_pred
df_misclassified = X_test.copy()
df_misclassified["true_label"] = y_test.values
df_misclassified["pred_label"] = y_pred
df_misclassified["true_label_name"] = df_misclassified["true_label"].map(mapping)
df_misclassified["pred_label_name"] = df_misclassified["pred_label"].map(mapping)

print(f"Total misclassified samples: {misclassified_mask.sum()} / {len(y_test)}")
print("\n--- Sample of Misclassified Cars ---\n")
if identifier_cols:
    cols = identifier_cols + ["true_label_name", "pred_label_name"] + numeric_features + categorical_features
else:
    cols = ["true_label_name", "pred_label_name"] + numeric_features + categorical_features

print(df_misclassified[cols].head(10).to_string(index=False))

df_misclassified.to_csv("plots/misclassified_samples.csv", index=False)
print("\nMisclassified sample details saved to 'plots/misclassified_samples.csv'")

# ============================================================
# STEP 6: Per-Cluster Error Breakdown
# ============================================================
error_summary = (
    df_misclassified.groupby("true_label_name")
    .size()
    .reset_index(name="Misclassified_Count")
    .sort_values(by="Misclassified_Count", ascending=False)
)
total_counts = y_test.value_counts().rename_axis("true_label_name").reset_index(name="Total_Count")
error_summary = pd.merge(error_summary, total_counts, on="true_label_name", how="right").fillna(0)
error_summary["Error_Rate_%"] = (error_summary["Misclassified_Count"] / error_summary["Total_Count"] * 100).round(2)

print("\n--- Per-Cluster Misclassification Summary ---\n")
print(error_summary.to_string(index=False))

plt.figure(figsize=(8, 5))
sns.barplot(x="true_label_name", y="Error_Rate_%", data=error_summary, palette="viridis")
plt.title("Per-Cluster Misclassification Rates")
plt.xlabel("Cluster Label")
plt.ylabel("Error Rate (%)")
plt.tight_layout()
plt.savefig("plots/per_cluster_error_rate.png")
plt.show()

# ============================================================
# STEP 7: Feature Correlation with Misclassification
# ============================================================
try:
    X_test_copy = X_test.copy()
    X_test_copy["error_flag"] = (y_test != y_pred).astype(int)
    corr = X_test_copy[numeric_features + ["error_flag"]].corr()["error_flag"].drop("error_flag")

    plt.figure(figsize=(7, 4))
    sns.barplot(x=corr.index, y=corr.values, palette="coolwarm")
    plt.title("Feature Correlation with Misclassification")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/feature_error_correlation.png")
    plt.show()
except Exception as e:
    print("Skipping correlation plot due to missing data:", e)

# ============================================================
# STEP 8: SHAP EXPLAINABILITY (Global + Local)
# ============================================================
print("\n--- SHAP EXPLAINABILITY SECTION ---\n")
try:
    # Extract classifier from the sklearn pipeline
    model_estimator = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']

    # Transform features for SHAP
    X_test_transformed = preprocessor.transform(X_test)

    # Initialize SHAP Explainer (TreeExplainer handles RF/XGBoost)
    explainer = shap.Explainer(model_estimator, X_test_transformed)
    shap_values = explainer(X_test_transformed)

    # --- Global Feature Importance ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, show=False)
    plt.title("SHAP Summary Plot (Global Feature Importance)")
    plt.tight_layout()
    plt.savefig("plots/shap_summary_global.png", bbox_inches='tight')
    plt.show()

    # --- Mean absolute SHAP values per feature (bar chart) ---
    plt.figure()
    shap.summary_plot(shap_values, X_test_transformed, plot_type="bar", show=False)
    plt.title("Mean SHAP Value (Feature Importance)")
    plt.tight_layout()
    plt.savefig("plots/shap_feature_importance_bar.png", bbox_inches='tight')
    plt.show()

    # --- Local Explanations for Top 5 Misclassified ---
    print("\nGenerating SHAP force plots for first 5 misclassified samples...")
    mis_idx = df_misclassified.index[:5]
    for i, idx in enumerate(mis_idx):
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)
        plt.title(f"Sample {i+1}: True={df_misclassified.loc[idx,'true_label_name']} | Pred={df_misclassified.loc[idx,'pred_label_name']}")
        plt.tight_layout()
        path = f"plots/shap_misclassified_{i+1}.png"
        plt.savefig(path, bbox_inches='tight')
        plt.close()
    print("Saved individual SHAP explanations for 5 misclassified samples.")
except Exception as e:
    print("⚠️ SHAP explainability failed or skipped:", e)

print("\n--- ERROR ANALYSIS + SHAP EXPLAINABILITY COMPLETE ---\n")
