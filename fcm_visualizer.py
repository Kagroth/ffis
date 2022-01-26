
from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np

class FCMVisualizer:
    def __init__(self, fcm_clustering_result=None) -> None:
        self.clustering_result = fcm_clustering_result
    
    def view_2d_partitions(self):
        fig, ax = plt.subplots(3, 3)
        x = y = 0

        for clustering_r in self.clustering_result:
            cluster_centers = clustering_r['cluster_centers']

            ax[x, y].plot(cluster_centers[:, 0], cluster_centers[:, 1], 'rs')
            
            for crisp_cluster in clustering_r['crisp_clusters']:
                clst = np.array(crisp_cluster)
                ax[x, y].scatter(clst[:, 0], clst[:, 1])
                ax[x, y].set_title("k = {}, fpc = {}".format(clustering_r['k'], round(clustering_r['fpc'], 3)))
            
            if y == 2:
                x += 1
                y = 0
            else:
                y += 1

    def view_3d_partitions(self):
        fig = plt.figure()
        # fig, ax = plt.subplots(3, 3)

        x = y = z = 1

        for clustering_r in self.clustering_result:
            cluster_centers = clustering_r['cluster_centers']
            
            ax = fig.add_subplot(3, 3, z, projection='3d')

            # ax[x, y].plot(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], 'rs')
            ax.plot(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], 'rs')
            
            for crisp_cluster in clustering_r['crisp_clusters']:
                clst = np.array(crisp_cluster)
                
                if len(crisp_cluster) == 0:
                    continue

                ax.scatter(clst[:, 0], clst[:, 1], clst[:, 2])
                ax.set_title("k = {}, fpc = {}".format(clustering_r['k'], round(clustering_r['fpc'], 3)))
            
            z += 1

            if y == 2:
                x += 1
                y = 0
            else:
                y += 1

    def view_all_partitions(self):
        dimensions = self.clustering_result[0]['cluster_centers'].shape[1]

        if dimensions == 2:
            self.view_2d_partitions()
        elif dimensions == 3:
            self.view_3d_partitions()
        else:
            print("Can't create plots for dimensionality of ", dimensions)
            return

        plt.tight_layout()
        plt.show()

    def view_membership(self, index):
        context_clustering_results = self.clustering_result[index]
        k = context_clustering_results['k']

        fig, ax = plt.subplots(1, k)

        x = 0
        for x in range(k):
            ax[x].plot(context_clustering_results['cluster_centers'][x, 0], context_clustering_results['cluster_centers'][x, 1], 'rs')
            ax[x].scatter(context_clustering_results['membership']['points'][0],
                         context_clustering_results['membership']['points'][1],
                         c=context_clustering_results['membership']['u'][x],
                         cmap="copper")
        plt.show()
