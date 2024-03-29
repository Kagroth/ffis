from tskmodel import TSKModel
from utils import create_fuzzy_variables_from_clusters, create_rules_from_clusters
from fcm_analyzer import FCMAnalyzer
from sklearn.model_selection import train_test_split

class LocalModel:
    def __init__(self, dataset, model_weight, feature_names, epochs=100, lr=0.2, momentum=0.9, eg=0.05, validity_method="Pearson") -> None:
        self.dataset = dataset
        self.model_weight = model_weight
        self.feature_names = feature_names
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.eg = eg
        self.fcm_analyzer = FCMAnalyzer(validity_method=validity_method)
        self.tsk_model = None
        self.scaler = None
        self.train_inputs = None
        self.train_outputs = None
        self.test_inputs = None
        self.test_outputs = None
        self.old_fis = None

    def set_fuzzy_inference_system(self, fis) -> None:
        self.old_fis = self.tsk_model.fis
        self.tsk_model.fis = fis

    def restore_local_fis(self) -> None:
        self.tsk_model.fis = self.old_fis

    def train_test_split(self) -> None:
        train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(self.scaled_data[:, :-1], self.scaled_data[:, -1], test_size=0.3)
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.test_inputs = test_inputs
        self.test_outputs = test_outputs

    def fcm(self, scaler) -> None:
        """
            Perform Fuzzy C-Means clustering using FCMAnalyzer object and store it in self.fcm_analyzer 
        """
        self.scaler = scaler
        scaled_data = self.scaler.fit_transform(self.dataset)
        self.scaled_data = scaled_data
        clustering_result = self.fcm_analyzer.fit(scaled_data.T, error=0.001, maxiter=100)

    def create_rules(self, rules_count=None) -> list:
        fuzzy_partition = None

        if rules_count is None:
            fuzzy_partition = self.fcm_analyzer.get_best_partition()
        else:
            fuzzy_partition = self.fcm_analyzer.get_partition(k=rules_count)

        fuzzy_vars = create_fuzzy_variables_from_clusters(fuzzy_partition['cluster_centers'], 
                                                cluster_stds=fuzzy_partition['crisp_cluster_stds'], 
                                                feature_names=self.feature_names, 
                                                show_fuzzy_vars=False)

        rules = create_rules_from_clusters(fuzzy_partition['cluster_centers'], fuzzy_vars)

        return rules

    def set_rules(self, rules) -> None:
        self.rules = rules

    def fit(self, rules=None) -> None:
        # third step - creation and training of TSK Model
        if rules is None:
            rules = self.rules
            
        self.tsk_model = TSKModel(rules, epochs=self.epochs, lr=self.lr, momentum=self.momentum, eg=self.eg)

        self.error_history = self.tsk_model.fit(self.train_inputs, self.feature_names[:-1], self.train_outputs)
        self.test_mse = self.tsk_model.test(self.test_inputs, self.test_outputs)
    
    def test(self) -> float:
        return self.tsk_model.test(self.test_inputs, self.test_outputs)