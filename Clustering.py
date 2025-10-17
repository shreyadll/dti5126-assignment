import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from cars_datasets import main as load_cars_dataset

# Load the preprocessed dataset
df_final = load_cars_dataset()
print("✅ Loaded cleaned car dataset.")
print(f"Shape: {df_final.shape}\n")
print("Columns:", df_final.columns.tolist())

# Select features for clustering
numeric_features = [
    'engine_displacement_in_cc',
    'battery_energy_capacity_in_kwh',
    'horsepower_in_hp',
    'torque_in_nm',
    'performance_0_to_100_km_per_h',
    'total_speed_in_km_per_h',
    'seats_parsed',
    'cars_price_amount'
]

categorical_features = [
    'fuel_types_normalized',
    'company_name_normalized'  # This will be replaced by 'company_segment' later
]

#  Ensure all features are numeric BEFORE calculation
# Convert numeric columns to numbers (coerce errors to NaN)
for col in numeric_features:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

# Replace placeholder values like -999 with NaN
df_final[numeric_features] = df_final[numeric_features].replace(-999, np.nan)

# FEATURE ENGINEERING (Value for Money)
# Value for Money is calculated as (Horsepower / Price) * 100,000.
# It measures performance efficiency relative to cost.

hp = df_final['horsepower_in_hp']
price = df_final['cars_price_amount']

# Impute NaNs for calculation (using median of non-NaN, non-zero prices/hp)
hp_safe = hp.fillna(hp[hp > 0].median())
price_safe = price.fillna(price[price > 0].median())

# Replace 0 prices with the median to prevent division by zero in the ratio
# Using .replace() ensures the original data type and index alignment is maintained
price_safe = price_safe.replace(0, price[price > 0].median() if price[price > 0].median() else 1)

# Calculate Value for Money (Horsepower / Price) and scale it up by 100,000
df_final['value_for_money'] = (hp_safe / price_safe) * 100000

# Add the new engineered feature to the list of numeric features for clustering
numeric_features.append('value_for_money')

# Handle missing values (numeric: fill with median)
X_numeric_filled = df_final[numeric_features].fillna(df_final[numeric_features].median())

# Display numeric feature distributions
print("\n--- Numeric Feature Distributions ---")
for col in numeric_features:
    print(
        f"{col}: min={X_numeric_filled[col].min()}, max={X_numeric_filled[col].max()}, "
        f"mean={X_numeric_filled[col].mean():.2f}")

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric_filled)

# CATEGORICAL FEATURE REFINEMENT (Brand Segmentation)

# Grouping brands into meaningful market segments for better interpretability
brand_groups = {
    'Luxury_Elite': ['Rolls Royce', 'Ferrari', 'Lamborghini', 'Bentley', 'Aston Martin', 'Mclaren'],
    'Premium_Performance': ['Mercedes', 'BMW', 'Audi', 'Porsche', 'Jaguar', 'Cadillac'],
    'Mass_Market_Mainstream': ['Ford', 'Mazda', 'Honda', 'Toyota', 'Volkswagen', 'Chevrolet', 'Nissan'],
    'Electric_Specialist': ['Tesla', 'Nio', 'Rivian'],
}


def map_brand_to_segment(company):
    company = str(company).strip()
    for segment, brands in brand_groups.items():
        if company in brands:
            return segment
    return 'Other_Market'


# Create the new segmented feature
df_final['company_segment'] = df_final['company_name_normalized'].apply(map_brand_to_segment)

# Update the categorical features list to use the segmented feature
categorical_features.remove('company_name_normalized')
categorical_features.append('company_segment')

# One-hot encode refined categorical features
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = encoder.fit_transform(df_final[categorical_features].fillna('Unknown'))

# Combine scaled numeric + OHE categorical
X_combined = np.hstack([X_scaled, X_encoded])
print(f"\nShape of initial feature matrix: {X_combined.shape}")

# OPTIMIZED DIMENSIONALITY REDUCTION (PCA for Clustering)
# Apply PCA to retain 90% of variance for clustering input (X_clustering)
pca_tuned = PCA(n_components=0.90, random_state=42)
X_clustering = pca_tuned.fit_transform(X_combined)

print(f"PCA Reduced Dimension to: {X_clustering.shape[1]} components")
print(f"Cumulative Explained Variance: {pca_tuned.explained_variance_ratio_.sum():.3f}")

# The first 2 components are used for visualization
X_pca_viz = X_clustering[:, 0:2]

# Tune DBSCAN parameters
# k-distance graph using the PCA-reduced data
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_clustering)
distances, indices = neighbors_fit.kneighbors(X_clustering)
distances = np.sort(distances[:, 4])  # 4th nearest neighbor
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.xlabel('Points sorted by distance (PCA-reduced)')
plt.ylabel('4th nearest neighbor distance')
plt.title('DBSCAN k-distance graph (PCA-reduced)')
plt.show()

# Automated DBSCAN tuning
eps_values = [0.1, 0.5, 1.0, 1.5, 2.0]  # Adjusted range for PCA data
min_samples_values = [3, 5, 7]

best_sil = -1
best_params = None

for eps in eps_values:
    for min_samples in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_clustering)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate silhouette only on non-noise points
        if n_clusters > 1 and np.sum(labels != -1) > 1:
            sil = silhouette_score(X_clustering[labels != -1], labels[labels != -1])
            if sil > best_sil:
                best_sil = sil
                best_params = (eps, min_samples)

if best_params:
    print(f"Best DBSCAN params: eps={best_params[0]}, min_samples={best_params[1]}, silhouette={best_sil:.3f}")
    dbscan_labels = DBSCAN(eps=best_params[0], min_samples=best_params[1]).fit_predict(X_clustering)
else:
    print("DBSCAN tuning failed to find a valid solution. Using fallback.")
    dbscan_labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(X_clustering)  # Fallback

df_final['DBSCAN_cluster'] = dbscan_labels

# Dimensionality reduction for visualization (using X_pca_viz)
print("\nExplained variance ratio by first 2 PCA components:", pca_tuned.explained_variance_ratio_[:2].sum())

# Plot PCA scatter of all data points
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_viz[:, 0], X_pca_viz[:, 1], s=30, alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("All Cars - PCA 2D Projection (Clustering Input)")
plt.show()

# Determine optimal number of clusters
K_range = range(2, 11)
wcss = []
sil_scores = []
best_k_stat = 0  # Variable to hold the statistically best k

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_clustering)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_clustering, labels))

# Plot Elbow Method
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'o-', color='blue')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method (PCA-reduced)')

# Plot Silhouette Scores
plt.subplot(1, 2, 2)
plt.plot(K_range, sil_scores, 'o-', color='green')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method (PCA-reduced)')
plt.tight_layout()
plt.show()

# selecting k with the highest silhouette score
best_k_stat = K_range[sil_scores.index(max(sil_scores))]
print(f"Optimal number of clusters based on silhouette score: {best_k_stat}")

# MANUALLY SET k FOR BETTER MARKET SEGMENTATION
# Based on project goals (Luxury, Performance, Budget, etc.),
# overriding the statistically optimal k=2 to k=4 for richer segmentation.
MANUAL_K_FOR_SEGMENTATION = 4
best_k = MANUAL_K_FOR_SEGMENTATION
print(f"-> Overriding k to {best_k} for K-Means/Agglomerative to meet business segmentation goals "
      f"(instead of {best_k_stat}).")

# Run clustering algorithms
kmeans_labels = KMeans(n_clusters=best_k, random_state=42, n_init='auto').fit_predict(X_clustering)
agglo_labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(X_clustering)

# Assign to dataframe
df_final['KMeans_cluster'] = kmeans_labels
df_final['Agglomerative_cluster'] = agglo_labels


# Evaluate clusters (Updated to handle DBSCAN noise)
def evaluate_clusters(X, labels, name):
    unique_labels = set(labels)
    non_noise_indices = labels != -1
    X_eval = X[non_noise_indices]
    labels_eval = labels[non_noise_indices]

    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

    # Need at least 2 clusters and 2 points to compute metrics
    if n_clusters > 1 and len(labels_eval) >= 2:
        sil = silhouette_score(X_eval, labels_eval)
        ch = calinski_harabasz_score(X_eval, labels_eval)
        return sil, ch, n_clusters
    else:
        return -1, -1, n_clusters


sil_kmeans, ch_kmeans, n_clusters_kmeans = evaluate_clusters(X_clustering, kmeans_labels, "KMeans")
sil_agglo, ch_agglo, n_clusters_agglo = evaluate_clusters(X_clustering, agglo_labels, "Agglomerative")
sil_dbscan, ch_dbscan, n_clusters_dbscan = evaluate_clusters(X_clustering, dbscan_labels, "DBSCAN")

print("\n--- Cluster Evaluation (Metrics on PCA-reduced data) ---")
print(f"KMeans: Silhouette Score = {sil_kmeans:.3f}, Calinski-Harabasz Score = {ch_kmeans:.3f}")
print(f"Agglomerative: Silhouette Score = {sil_agglo:.3f}, Calinski-Harabasz Score = {ch_agglo:.3f}")
print(f"DBSCAN: Silhouette Score = {sil_dbscan:.3f}, Calinski-Harabasz Score = {ch_dbscan:.3f}")

# Recommend the best clustering algorithm
metrics = {
    'KMeans': {'Silhouette': sil_kmeans, 'Calinski-Harabasz': ch_kmeans, 'Labels': kmeans_labels,
               'Clusters': n_clusters_kmeans},
    'Agglomerative': {'Silhouette': sil_agglo, 'Calinski-Harabasz': ch_agglo, 'Labels': agglo_labels,
                      'Clusters': n_clusters_agglo},
    'DBSCAN': {'Silhouette': sil_dbscan, 'Calinski-Harabasz': ch_dbscan, 'Labels': dbscan_labels,
               'Clusters': n_clusters_dbscan}
}

# Pick the algorithm with the highest silhouette score (tie-breaker: Calinski-Harabasz)
best_algo = max(metrics.items(), key=lambda x: (x[1]['Silhouette'], x[1]['Calinski-Harabasz']))

print("\n🎯 Recommended clustering algorithm:")
print(f"{best_algo[0]} (Silhouette={best_algo[1]['Silhouette']:.3f}, "
      f"Calinski-Harabasz={best_algo[1]['Calinski-Harabasz']:.3f})")

# Visualize clusters (2D PCA)
df_pca = pd.DataFrame(X_pca_viz, columns=['PC1', 'PC2'])
df_pca['KMeans'] = kmeans_labels
df_pca['Agglomerative'] = agglo_labels
df_pca['DBSCAN'] = dbscan_labels

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x='PC1', y='PC2', hue='KMeans', data=df_pca, palette='tab10', s=50)
plt.title(f'KMeans Clusters (k={best_k})')

plt.subplot(1, 3, 2)
sns.scatterplot(x='PC1', y='PC2', hue='Agglomerative', data=df_pca, palette='tab10', s=50)
plt.title(f'Agglomerative Clusters (k={best_k})')

# DBSCAN visualization
plt.subplot(1, 3, 3)
unique_labels = set(dbscan_labels)
n_clusters_dbscan_viz = len(unique_labels) - (1 if -1 in unique_labels else 0)

# Generate a palette for clusters (excluding noise)
palette = sns.color_palette("tab10", n_colors=max(n_clusters_dbscan_viz, 1))
colors = []
for lbl in dbscan_labels:
    if lbl == -1:
        colors.append('gray')  # noise
    elif n_clusters_dbscan_viz > 0:
        colors.append(palette[lbl % n_clusters_dbscan_viz])
    else:
        colors.append('black')

plt.scatter(df_pca['PC1'], df_pca['PC2'], c=colors, s=50)
plt.title(f'DBSCAN Clusters ({n_clusters_dbscan_viz} clusters)')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()

# Cluster Interpretation / Profiling
if best_algo[0] != 'None' and best_algo[1]['Silhouette'] > -1:
    df_final['Final_Cluster'] = best_algo[1]['Labels']

    # Define a core set of features for interpretation
    profile_features = ['cars_price_amount', 'horsepower_in_hp', 'value_for_money',
                        'performance_0_to_100_km_per_h', 'seats_parsed']

    print(f"\n--- Final Cluster Profiles ({best_algo[0]}) ---")

    # Numeric Feature Analysis: Mean/Median (Transposed)
    profile_numeric = df_final.groupby('Final_Cluster')[profile_features].agg(['mean', 'median']).round(2).T
    print("\nProfile: Mean/Median of Numeric Features per Cluster (Transposed)")
    print(profile_numeric)

    # Categorical Feature Analysis: Mode (Most Frequent) (Transposed)
    profile_categorical = df_final.groupby('Final_Cluster')[['fuel_types_normalized', 'company_segment']].agg(
        lambda x: x.mode()[0]).T
    print("\nProfile: Most Frequent Category (Mode) per Cluster (Transposed)")
    print(profile_categorical)
else:
    print("\nCannot perform cluster profiling: No valid clusters were found by the best algorithm.")

# Inspect sample clustered data
print("\n--- Sample clustered dataset ---")
# Use the updated categorical features list
sample_cols = [c for c in numeric_features + ['fuel_types_normalized', 'company_segment'] +
               ['KMeans_cluster', 'Agglomerative_cluster', 'DBSCAN_cluster'] if c in df_final.columns]
print(df_final.head(10)[sample_cols])

# Summary Table of Clustering Metrics
summary = []

# Recalculate summary using the robust evaluate_clusters results
for name, data in metrics.items():
    summary.append({
        'Algorithm': name,
        'Clusters': data['Clusters'],
        'Silhouette Score': round(data['Silhouette'], 3),
        'Calinski-Harabasz Score': round(data['Calinski-Harabasz'], 3)
    })

summary_df = pd.DataFrame(summary)
print("\n📊 Clustering Summary Table")
print(summary_df)
