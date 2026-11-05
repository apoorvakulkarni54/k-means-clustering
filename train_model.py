"""
train_model.py  -  Trains the K-Means model and saves it.
Run once:  python train_model.py
Steps follow the ML lifecycle: load -> understand -> clean -> select features
-> scale -> choose K -> train -> evaluate -> save.
"""
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["energy_consumption", "temperature"]   # numeric features for clustering

# 1. Data collection
df = pd.read_csv("data/energy_data.csv")

# 2. Preprocessing: remove duplicates and missing rows
df = df.drop_duplicates().dropna(subset=FEATURES)

# 3. Feature scaling (K-Means uses distance, so features need the same scale)
scaler = StandardScaler()
X = scaler.fit_transform(df[FEATURES])

# 4. Choose K: try K = 2..10 and keep WCSS + silhouette for each
results = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    results.append({"k": k, "wcss": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
results = pd.DataFrame(results)
print(results.round(4).to_string(index=False))

# Best K = highest silhouette score
best_k = int(results.loc[results["silhouette"].idxmax(), "k"])
print("\nChosen K:", best_k)

# 5. Train final model
model = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
df["cluster"] = model.labels_

# 6. Name clusters by their average energy use (lowest -> highest)
order = df.groupby("cluster")["energy_consumption"].mean().sort_values().index.tolist()
profile = df.groupby("cluster")[FEATURES].mean().round(2)
profile["size"] = df["cluster"].value_counts()
print("\nCluster profile:\n", profile)

# 7. Save everything the web app needs
joblib.dump(
    {"model": model, "scaler": scaler, "features": FEATURES,
     "best_k": best_k, "metrics": results, "rank_order": order},
    "model/kmeans_model.joblib",
)
df.to_csv("data/energy_clustered.csv", index=False)
print("\nSaved model/kmeans_model.joblib")
