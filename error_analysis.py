import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

PLOTS_DIR = "plots/error_analysis"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 1) Load clustered dataset + trained pipeline (preprocessor+model)
# ------------------------------------------------------------
df = pd.read_csv("clustered_cars_data.csv")
pipeline = joblib.load("models/best_classification_model.joblib")
reverse_mapping = joblib.load("models/reverse_mapping.joblib")

# Feature lists must match training-time definitions
numeric_features = [
    'engine_displacement_in_cc','battery_energy_capacity_in_kwh',
    'horsepower_in_hp','torque_in_nm','performance_0_to_100_km_per_h',
    'total_speed_in_km_per_h','seats_parsed','cars_price_amount',
    'value_for_money'
]
categorical_features = ['fuel_types_normalized','company_segment']
target = 'Final_Cluster'

# Keep a couple of human-readable columns if present (nice for tables)
id_cols = [c for c in ['company_name','car_name'] if c in df.columns]

# Prepare X/y using the same columns as training
X = df[numeric_features + categorical_features].copy()
y = df[target].astype(int).copy()

# Basic imputations (same simple approach as training)
X[numeric_features] = X[numeric_features].fillna(X[numeric_features].median())
X[categorical_features] = X[categorical_features].fillna('Unknown')

# Hold-out split to evaluate errors on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Predict labels with the trained pipeline
y_pred = pipeline.predict(X_test)

# ====================================================================
#1) Confusion Matrix (Counts)
#    Purpose: See which clusters the model confuses most often.
# ====================================================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Counts) — Hold-out Test")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cm_counts_simple.png"))
plt.close()

# ====================================================================
# 2) Misclassification Rate by Cluster
#    Purpose: Identify clusters that are harder than others.
# ====================================================================
err_df = pd.DataFrame({"true": y_test, "pred": y_pred})
err_df["error_flag"] = (err_df["true"] != err_df["pred"]).astype(int)
rate = (err_df.groupby("true")["error_flag"].mean() * 100).sort_index()

plt.figure(figsize=(7,4))
rate.plot(kind="bar", color="teal")
plt.ylabel("Error Rate (%)")
plt.title("Misclassification Rate by Cluster (Hold-out Test)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cluster_error_rates_simple.png"))
plt.close()

# ====================================================================
#  3) Misclassified Samples Table
#    Purpose: Inspect wrong predictions (useful for qualitative review).
# ====================================================================
mis_idx = np.where(y_test != y_pred)[0]
mis_table = X_test.iloc[mis_idx].copy()
if id_cols:
    # Attach identifiers from original df for easier human reading
    try:
        mis_table[id_cols] = df.loc[X_test.index[mis_idx], id_cols]
    except Exception:
        pass

# Map integer labels back to original cluster ids for readability
mis_table["True_Label"] = y_test.iloc[mis_idx].map(reverse_mapping)
mis_table["Pred_Label"] = pd.Series(y_pred, index=X_test.index).iloc[mis_idx].map(reverse_mapping)

mis_csv = os.path.join(PLOTS_DIR, "misclassified_samples_simple.csv")
mis_table.to_csv(mis_csv, index=False)

# ====================================================================
#  4) Feature Distributions: Correct vs Misclassified (Boxplots)
#    Purpose: Spot feature ranges where mistakes are common.
# ====================================================================
flag_df = X_test.copy()
flag_df["error_flag"] = (y_test != y_pred).astype(int)

for col in numeric_features:
    plt.figure(figsize=(7,4))
    sns.boxplot(x="error_flag", y=col, data=flag_df)
    plt.title(f"{col}: Correct (0) vs Misclassified (1)")
    plt.xlabel("Error Flag")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"box_{col}_simple.png"))
    plt.close()

# ====================================================================
# 5) Feature Importance Bar Chart 
#    Purpose: Which features matter most to the trained classifier?
#    Notes:
#      - We extract the underlying tree model from the pipeline.
#      - We expand categorical feature names using the encoder, so
#        importances correspond to actual transformed columns.
#      - If the classifier lacks feature_importances_, we skip gracefully.
# ====================================================================
try:
    preproc = pipeline.named_steps['preprocessor']
    clf     = pipeline.named_steps['classifier']

    # 1) Build transformed feature names:
    #    - numeric names are unchanged by the scaler
    #    - categorical names are expanded by OneHotEncoder
    num_names = numeric_features
    cat_enc = preproc.named_transformers_['cat'].named_steps['encoder']
    cat_names = list(cat_enc.get_feature_names_out(categorical_features))
    feature_names = num_names + cat_names

    # 2) Extract feature importances from the tree-based model (RF/XGB)
    if hasattr(clf, "feature_importances_"):
        importances = pd.Series(clf.feature_importances_, index=feature_names)
        top = importances.sort_values(ascending=False).head(20)

        plt.figure(figsize=(8,6))
        top.sort_values().plot(kind="barh", color="slateblue")
        plt.title("Top Feature Importances (model.feature_importances_)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importances_simple.png"))
        plt.close()
    else:
        # If the classifier doesn't support importances, leave a note
        with open(os.path.join(PLOTS_DIR, "feature_importances_simple.txt"), "w") as f:
            f.write("Classifier does not expose feature_importances_. Skipped.\n")
except Exception as e:
    # Avoid crashing the whole analysis if importance visualization fails
    with open(os.path.join(PLOTS_DIR, "feature_importances_simple_error.txt"), "w") as f:
        f.write(f"Feature importance plot skipped due to: {e}\n")

print("\n Simplified error analysis (no SHAP) completed. Outputs saved to 'plots/'.\n")
