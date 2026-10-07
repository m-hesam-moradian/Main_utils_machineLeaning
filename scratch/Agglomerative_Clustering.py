import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score, mutual_info_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')
import os

# Set file path
data_path = r"d:\ML\task\Data.xlsx"

# Read data
xls = pd.ExcelFile(data_path)
sheets = ['D1', 'D2', 'D3', 'D4']

# Dictionary to store all results
all_reports = {}
all_clustered_data = {}

def calc_entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return entropy(probs)

def calc_variation_of_information(labels_true, labels_pred):
    h_true = calc_entropy(labels_true)
    h_pred = calc_entropy(labels_pred)
    mi = mutual_info_score(labels_true, labels_pred)
    return h_true + h_pred - 2 * mi

# Configuration
min_k = 2
max_k = 10
n_runs = 5
linkage_method = 'ward'

for sheet in sheets:
    print(f"Processing feature set: {sheet}...")
    df = pd.read_excel(xls, sheet_name=sheet)
    
    # 1. Extract features (exclude ID)
    features = df.drop(columns=['ID']).copy()
    feature_names = features.columns.tolist()
    
    # 2. Apply z-score standardization
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine the final number of clusters using Silhouette score
    best_k = min_k
    best_silhouette = -1
    for k in range(min_k, max_k + 1):
        clusterer = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = clusterer.fit_predict(features_scaled)
        score = silhouette_score(features_scaled, labels)
        if score > best_silhouette:
            best_silhouette = score
            best_k = k
            
    print(f"[{sheet}] Optimal number of clusters chosen: {best_k} (Silhouette: {best_silhouette:.4f})")
    
    # Run 5 independent runs
    run_labels = []
    run_reports = []
    
    for run in range(n_runs):
        # To simulate independent runs for deterministic Agglomerative Clustering,
        # we inject a very small amount of noise (e.g., 1e-4) which won't change the underlying structure
        # but may alter distance ties.
        np.random.seed(42 + run)
        noise = np.random.normal(0, 1e-4, size=features_scaled.shape)
        features_noisy = features_scaled + noise
        
        # 3. Construct pairwise distance matrix (implicitly done by sklearn in fit)
        # 4 & 6. Apply Agglomerative Clustering & Assign
        clusterer = AgglomerativeClustering(n_clusters=best_k, linkage=linkage_method)
        labels = clusterer.fit_predict(features_noisy)
        run_labels.append(labels)
        
        # --- Run Level Reporting ---
        # Number of observations per cluster
        unique, counts = np.unique(labels, return_counts=True)
        size_dist = dict(zip(unique, counts))
        proportions = {k: v/len(labels) for k, v in size_dist.items()}
        
        # Calculate within-cluster and between-cluster distance
        # We compute the full distance matrix for evaluation
        dist_matrix = squareform(pdist(features_noisy, metric='euclidean'))
        
        within_dists = []
        for c in unique:
            idx = np.where(labels == c)[0]
            if len(idx) > 1:
                # average pairwise distance within the cluster
                sub_dist = dist_matrix[np.ix_(idx, idx)]
                avg_within = sub_dist[np.triu_indices_from(sub_dist, k=1)].mean()
                within_dists.append(avg_within)
            else:
                within_dists.append(0)
                
        mean_within_cluster_dist = np.mean(within_dists)
        
        between_dists = []
        for c1 in unique:
            for c2 in unique:
                if c1 < c2:
                    idx1 = np.where(labels == c1)[0]
                    idx2 = np.where(labels == c2)[0]
                    sub_dist = dist_matrix[np.ix_(idx1, idx2)]
                    between_dists.append(sub_dist.mean())
                    
        mean_between_cluster_dist = np.mean(between_dists) if between_dists else 0
        
        # Feature-wise mean and std (centroid profile)
        cluster_profiles = []
        for c in unique:
            idx = np.where(labels == c)[0]
            c_mean = features.iloc[idx].mean()
            c_std = features.iloc[idx].std()
            profile = {
                "Run": run + 1,
                "Cluster": c,
                "Size": size_dist[c],
                "Proportion (%)": proportions[c]*100,
            }
            for col in feature_names:
                profile[f"{col}_mean"] = c_mean[col]
                profile[f"{col}_std"] = c_std[col]
            cluster_profiles.append(profile)
            
        run_reports.append({
            "Run": run + 1,
            "Clusters": best_k,
            "Mean_Within_Cluster_Dist": mean_within_cluster_dist,
            "Mean_Between_Cluster_Dist": mean_between_cluster_dist,
            "Silhouette_Score": silhouette_score(features_noisy, labels)
        })
        
        # Add labels to dataframe
        df[f'Cluster_Run_{run+1}'] = labels
        
    all_clustered_data[f"{sheet}_Clustered"] = df
    
    # Create comparison matrices
    ari_matrix = np.zeros((n_runs, n_runs))
    ami_matrix = np.zeros((n_runs, n_runs))
    vi_matrix = np.zeros((n_runs, n_runs))
    
    for i in range(n_runs):
        for j in range(n_runs):
            ari_matrix[i, j] = adjusted_rand_score(run_labels[i], run_labels[j])
            ami_matrix[i, j] = adjusted_mutual_info_score(run_labels[i], run_labels[j])
            vi_matrix[i, j] = calc_variation_of_information(run_labels[i], run_labels[j])
            
    # Combine run reports
    df_run_summary = pd.DataFrame(run_reports)
    df_profiles = pd.DataFrame(cluster_profiles)
    
    df_ari = pd.DataFrame(ari_matrix, columns=[f"Run_{x+1}" for x in range(n_runs)], index=[f"Run_{x+1}" for x in range(n_runs)])
    df_ami = pd.DataFrame(ami_matrix, columns=[f"Run_{x+1}" for x in range(n_runs)], index=[f"Run_{x+1}" for x in range(n_runs)])
    df_vi = pd.DataFrame(vi_matrix, columns=[f"Run_{x+1}" for x in range(n_runs)], index=[f"Run_{x+1}" for x in range(n_runs)])
    
    # Store all reports for the sheet
    report_frames = {
        "Run_Summary": df_run_summary,
        "Cluster_Profiles": df_profiles,
        "ARI_Matrix": df_ari,
        "AMI_Matrix": df_ami,
        "VI_Matrix": df_vi
    }
    all_reports[f"{sheet}_Report"] = report_frames

# Save everything to the existing Data.xlsx
print("Saving results back to Excel...")
try:
    with pd.ExcelWriter(data_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        for sheet_name, df_clustered in all_clustered_data.items():
            df_clustered.to_excel(writer, sheet_name=sheet_name, index=False)
            
        for report_name, frames in all_reports.items():
            # We will concatenate the frames into one sheet to save space, or just write them to separate columns
            start_row = 0
            for title, df_frame in frames.items():
                # Write title
                pd.DataFrame([[title]]).to_excel(writer, sheet_name=report_name, startrow=start_row, startcol=0, index=False, header=False)
                start_row += 1
                # Write data
                df_frame.to_excel(writer, sheet_name=report_name, startrow=start_row, startcol=0, index=True)
                start_row += len(df_frame) + 3
except Exception as e:
    print(f"Error saving to Excel: {e}")
    # In case Data.xlsx is locked, save to a fallback Clustering_Report.xlsx
    fallback_path = r"d:\ML\task\Clustering_Report.xlsx"
    print(f"Saving to fallback {fallback_path}")
    with pd.ExcelWriter(fallback_path, engine='openpyxl') as writer:
        for sheet_name, df_clustered in all_clustered_data.items():
            df_clustered.to_excel(writer, sheet_name=sheet_name, index=False)
        for report_name, frames in all_reports.items():
            start_row = 0
            for title, df_frame in frames.items():
                pd.DataFrame([[title]]).to_excel(writer, sheet_name=report_name, startrow=start_row, startcol=0, index=False, header=False)
                start_row += 1
                df_frame.to_excel(writer, sheet_name=report_name, startrow=start_row, startcol=0, index=True)
                start_row += len(df_frame) + 3

print("Process completed successfully.")
