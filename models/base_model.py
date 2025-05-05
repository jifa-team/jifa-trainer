from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import confusion_matrix

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @abstractmethod
    def train(self, X_train, y_train):
        """Treina o modelo com os dados fornecidos"""
        pass

    @abstractmethod
    def predict(self, X):
        """Faz previsões com o modelo treinado"""
        pass

    def evaluate(self):
        """Avalia o modelo e retorna métricas"""
        if self.model is None:
            raise ValueError("Modelo não treinado")

        # Métricas de teste
        y_pred_test = self.predict(self.X_test)
        acc_t, prec_t, rec_t = self._calculate_metrics(self.y_test, y_pred_test)
        metrics_test = {"accuracy": acc_t, "precision": prec_t, "recall": rec_t}

        # Métricas de treino
        y_pred_train = self.predict(self.X_train)
        acc_tr, prec_tr, rec_tr = self._calculate_metrics(self.y_train, y_pred_train)
        metrics_train = {"accuracy": acc_tr, "precision": prec_tr, "recall": rec_tr}

        return {
            "test_metrics": metrics_test,
            "train_metrics": metrics_train
        }

    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas manualmente"""
        cm = confusion_matrix(y_true, y_pred)
        supports = cm.sum(axis=1)
        precisions = [(cm[i,i] / cm[:,i].sum()) if cm[:,i].sum()>0 else 0 for i in range(len(cm))]
        recalls = [(cm[i,i] / cm[i,:].sum()) if cm[i,:].sum()>0 else 0 for i in range(len(cm))]
        acc = np.trace(cm) / cm.sum()
        precision = np.average(precisions, weights=supports)
        recall = np.average(recalls, weights=supports)
        return round(acc,3), round(precision,3), round(recall,3)

    def get_confusion_matrix(self):
        """Retorna a matriz de confusão"""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        y_pred = self.predict(self.X_test)
        return confusion_matrix(self.y_test, y_pred)

    def get_decision_surface(self, feature_indices=[2,3]):
        """Retorna os dados para plotar a superfície de decisão"""
        if self.model is None:
            raise ValueError("Modelo não treinado")
        
        # Seleciona apenas as features especificadas
        X2_train = self.X_train[:,feature_indices]
        X2_test = self.X_test[:,feature_indices]
        
        # Treina um modelo auxiliar apenas com as features selecionadas
        model2 = self.__class__()
        model2.train(X2_train, self.y_train)
        
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
            'y_test': self.y_test
        } 