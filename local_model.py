from operator import imod
from tskmodel import TSKModel
from utils import create_fuzzy_variables_from_clusters, create_rules_from_clusters
from fcm_analyzer import FCMAnalyzer
from sklearn.model_selection import train_test_split

class LocalModel:
    def __init__(self, dataset, feature_names, epochs=100, lr=0.2, momentum=0.9, eg=0.05) -> None:
        self.dataset = dataset
        self.feature_names = feature_names
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.eg = eg
        self.fcm_analyzer = FCMAnalyzer()
        self.tsk_model = None
        self.scaler = None
        self.test_inputs = None
        self.test_outputs = None

    def set_fuzzy_inference_system(self, fis) -> None:
        self.tsk_model.fis = fis

    def fit(self, scaler, rules_count=2) -> None:
        # first step - normalize the data
        self.scaler = scaler
        scaled_data = self.scaler.fit_transform(self.dataset)
        train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(scaled_data[:, :-1], scaled_data[:, -1], test_size=0.2)
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

        # second step - fuzzy c-means and rules creation
        clustering_result = self.fcm_analyzer.fit(scaled_data.T, error=0.001, maxiter=100)
        fuzzy_partition = self.fcm_analyzer.get_partition(k=rules_count)
        fuzzy_vars = create_fuzzy_variables_from_clusters(fuzzy_partition['cluster_centers'], 
                                                cluster_stds=fuzzy_partition['crisp_cluster_stds'], 
                                                feature_names=self.feature_names, 
                                                show_fuzzy_vars=False)

        rules = create_rules_from_clusters(fuzzy_partition['cluster_centers'], fuzzy_vars)

        # third step - creation and training of TSK Model
        self.tsk_model = TSKModel(rules, epochs=self.epochs, lr=self.lr, momentum=self.momentum, eg=self.eg)

        self.error_history = self.tsk_model.fit(train_inputs, self.feature_names[:-1], train_outputs)
        self.test_mse = self.tsk_model.test(test_inputs, test_outputs)
    
    def test(self) -> float:
        return self.tsk_model.test(self.test_inputs, self.test_outputs)