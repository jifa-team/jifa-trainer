from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
from .base_model import BaseModel

class KMeansModel(BaseModel):
    def __init__(self, n_clusters=3, random_state=42):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None

    def train(self, X_train, y_train):
        """Treina o modelo K-Means"""
        self.X_train = X_train
        self.y_train = y_train
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.model.fit(X_train)
        self.cluster_centers_ = self.model.cluster_centers_

    def predict(self, X):
        """Faz previsões com o modelo K-Means"""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        return self.model.predict(X)

    def set_test_data(self, X_test, y_test):
        """Define os dados de teste"""
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """Avalia o modelo e retorna métricas específicas do K-Means"""
        if self.model is None:
            raise ValueError("Modelo não treinado")

        # Métricas de treino
        y_pred_train = self.predict(self.X_train)
        silhouette_train = silhouette_score(self.X_train, y_pred_train)
        calinski_train = calinski_harabasz_score(self.X_train, y_pred_train)
        davies_train = davies_bouldin_score(self.X_train, y_pred_train)

        # Métricas de teste
        y_pred_test = self.predict(self.X_test)
        silhouette_test = silhouette_score(self.X_test, y_pred_test)
        calinski_test = calinski_harabasz_score(self.X_test, y_pred_test)
        davies_test = davies_bouldin_score(self.X_test, y_pred_test)

        return {
            "test_metrics": {
                "silhouette": round(silhouette_test, 3),
                "calinski_harabasz": round(calinski_test, 3),
                "davies_bouldin": round(davies_test, 3)
            },
            "train_metrics": {
                "silhouette": round(silhouette_train, 3),
                "calinski_harabasz": round(calinski_train, 3),
                "davies_bouldin": round(davies_train, 3)
            }
        }

    def get_decision_surface(self, feature_indices=[2,3]):
        """Retorna os dados para plotar a superfície de decisão"""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        
        # Seleciona apenas as features especificadas
        X2_train = self.X_train[:,feature_indices]
        X2_test = self.X_test[:,feature_indices]
        
        # Treina um modelo auxiliar apenas com as features selecionadas
        model2 = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        model2.fit(X2_train)
        
        # Gera a grade para plotagem
        x_min, x_max = X2_train[:,0].min()-1, X2_train[:,0].max()+1
        y_min, y_max = X2_train[:,1].min()-1, X2_train[:,1].max()+1
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )
        
        # Faz previsões para a grade
        Z = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        return {
            'xx': xx,
            'yy': yy,
            'Z': Z,
            'X_test': X2_test,
            'y_test': self.y_test,
            'centers': model2.cluster_centers_
        } 