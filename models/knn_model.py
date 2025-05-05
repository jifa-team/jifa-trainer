from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self, n_neighbors=3):
        super().__init__()
        self.n_neighbors = n_neighbors

    def train(self, X_train, y_train):
        """Treina o modelo KNN"""
        self.X_train = X_train
        self.y_train = y_train
        self.model = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Faz previsões com o modelo KNN"""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        return self.model.predict(X)

    def set_test_data(self, X_test, y_test):
        """Define os dados de teste"""
        self.X_test = X_test
        self.y_test = y_test 