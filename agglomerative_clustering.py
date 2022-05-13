import numpy as np
from sklearn.metrics import pairwise_distances

def agglomerative_clustering(cluster_centers: list, cluster_stds: list, threshold: float):
    merged_clusters = list()
    merged_stds = list()
    models_count = len(cluster_centers)
    iter_counter = 0

    while True:
        # calculate distance matrices between clusters centers from different models
        minimum_distances = list()
        for i in range(models_count):
            for j in range(models_count):
                if i >= j:
                    continue

                if cluster_centers[i].shape[0] == 0 or cluster_centers[j].shape[0] == 0:
                    # if there is no cluster center in model, then don't create distance matrix 
                    continue

                # print("Distance matrix between clusters of models {} and {}".format(i + 1, j + 1))
                dmat = pairwise_distances(cluster_centers[i], cluster_centers[j])
                # print(dmat)
                # print("Wartosc min: {}".format(np.amin(dmat)))
                index = np.unravel_index(np.argmin(dmat), dmat.shape)
                # index = np.where(dmat == np.amin(dmat))
                # print("Indeks minimalnej wartosci: {}".format(index))
                # print()
                minimum_distances.append({
                    "model_indices": (i, j),
                    "cluster_indices": index,
                    "cluster_distance": np.amin(dmat)
                })
            
        print("Minimum distances list of dicts: ", minimum_distances)
        # find minimum distance
        if len(minimum_distances) == 0:
            # no distance matrix created - end of clustering:
            print("End of agglomerative clustering")
            print("Merged clusters count: ", len(merged_clusters))
            # print("Merged clusters: ", merged_clusters)
            return merged_clusters, merged_stds

        entry_with_min_d = minimum_distances[0]
        for distance_entry in minimum_distances:
            if entry_with_min_d["cluster_distance"] > distance_entry["cluster_distance"]:
                entry_with_min_d = distance_entry

        print(entry_with_min_d)
        local_clusters_left = sum([local_clusters.shape[0] for local_clusters in cluster_centers]) 
        print("Total clusters count: ", local_clusters_left)

        # compare with threshold
        if entry_with_min_d["cluster_distance"] <= threshold:
            # merge clusters and remove them from local lists
            print("Min is {} and is lower than threshold {}".format(entry_with_min_d["cluster_distance"], threshold))
            first_model_index, second_model_index = entry_with_min_d["model_indices"]
            first_cluster_index, second_cluster_index = entry_with_min_d["cluster_indices"]
            print("Clusters to be merged:")
            first_cluster = cluster_centers[first_model_index][first_cluster_index, :]
            second_cluster = cluster_centers[second_model_index][second_cluster_index, :]
            merged_cluster = (first_cluster + second_cluster) / 2.0
            print("Cluster {} from model {}: {}".format(first_cluster_index, first_model_index, first_cluster))
            print("and")
            print("Cluster {} from model {}: {}".format(second_cluster_index, second_model_index, second_cluster))
            # print("Merged cluster: {}".format(merged_cluster))
            merged_clusters.append(merged_cluster)
            cluster_centers[first_model_index] = np.delete(cluster_centers[first_model_index], first_cluster_index, 0)
            cluster_centers[second_model_index] = np.delete(cluster_centers[second_model_index], second_cluster_index, 0)

            # merge stds
            first_cluster_std = cluster_stds[first_model_index][first_cluster_index, :]
            second_cluster_std = cluster_stds[second_model_index][second_cluster_index, :]
            merged_std = (first_cluster_std + second_cluster_std) / 2.0
            print("Cluster std {} from model {}: {}".format(first_cluster_index, first_model_index, first_cluster_std))
            print("and")
            print("Cluster std {} from model {}: {}".format(second_cluster_index, second_model_index, second_cluster_std))
            merged_stds.append(merged_std)
            cluster_stds[first_model_index] = np.delete(cluster_stds[first_model_index], first_cluster_index, 0)
            cluster_stds[second_model_index] = np.delete(cluster_stds[second_model_index], second_cluster_index, 0)
        elif len(merged_clusters) < 2:
            # if there is no merged clusters and minimal distance is higher than threshold, then increase threshold and continue clustering
            print("Minimal distance below threshold and there is no merged clusters: threshold increased by 0.2")
            threshold += 0.2
        else:
            # return merged clusters
            print("Min is {} and it is not lower than threshold {}".format(entry_with_min_d["cluster_distance"], threshold))
            print("End of agglomerative clustering")
            print("Merged clusters count: ", len(merged_clusters))
            # print("Merged clusters: ", merged_clusters)
            return merged_clusters, merged_stds

        # if sum(clusters len) > 1 then go to calc distance
        local_clusters_left = sum([local_clusters.shape[0] for local_clusters in cluster_centers]) 
        print("Local clusters left: ", local_clusters_left)

        if local_clusters_left <= 1:
            # return merged clusters
            print("End of agglomerative clustering")
            print("Merged clusters count: ", len(merged_clusters))
            # print("Merged clusters: ", merged_clusters)
            return merged_clusters, merged_stds

        iter_counter += 1
        print("Iteration count: ", iter_counter)