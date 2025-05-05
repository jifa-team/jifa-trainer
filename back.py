import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend sem GUI
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

# Variáveis globais para modelo e dados
model = None
X_train = X_test = y_train = y_test = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classifier')
def classifier():
    model_type = request.args.get('model', 'knn')
    return render_template('front.html', model_type=model_type)

@app.route('/train', methods=['POST'])
def train():
    global model, X_train, X_test, y_train, y_test
    model_type = request.args.get('model', 'knn')
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )
    
    if model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=3)
    # Adicione outros modelos aqui no futuro
    
    model.fit(X_train, y_train)
    return jsonify({"message": "Treinamento concluído"})

@app.route('/test', methods=['GET'])
def test():
    if model is None:
        return jsonify({"error": "Modelo não treinado"}), 400

    # Função auxiliar para calcular métricas manualmente
    def calc_manual(cm):
        supports = cm.sum(axis=1)
        precisions = [ (cm[i,i] / cm[:,i].sum()) if cm[:,i].sum()>0 else 0 for i in range(len(cm)) ]
        recalls = [ (cm[i,i] / cm[i,:].sum()) if cm[i,:].sum()>0 else 0 for i in range(len(cm)) ]
        acc = np.trace(cm) / cm.sum()
        precision = np.average(precisions, weights=supports)
        recall = np.average(recalls, weights=supports)
        return round(acc,3), round(precision,3), round(recall,3)

    # Test metrics
    y_pred_test = model.predict(X_test)
    acc_t, prec_t, rec_t = calc_manual(confusion_matrix(y_test, y_pred_test))
    metrics_test = {"accuracy": acc_t, "precision": prec_t, "recall": rec_t}

    # Train metrics
    y_pred_train = model.predict(X_train)
    acc_tr, prec_tr, rec_tr = calc_manual(confusion_matrix(y_train, y_pred_train))
    metrics_train = {"accuracy": acc_tr, "precision": prec_tr, "recall": rec_tr}

    # Gera imagens (base64)
    def gen_confusion_image(cm):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        ticks = np.arange(len(load_iris().target_names))
        plt.xticks(ticks, load_iris().target_names, rotation=45)
        plt.yticks(ticks, load_iris().target_names)
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        buf = io.BytesIO(); plt.tight_layout(); plt.savefig(buf, format='png'); buf.seek(0); plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    cm_img = gen_confusion_image(confusion_matrix(y_test, y_pred_test))

    # Superfície de decisão (atributos 2 e 3)
    X2_train = X_train[:,2:4]
    X2_test = X_test[:,2:4]
    model2 = KNeighborsClassifier(n_neighbors=3).fit(X2_train, y_train)
    x_min, x_max = X2_train[:,0].min()-1, X2_train[:,0].max()+1
    y_min, y_max = X2_train[:,1].min()-1, X2_train[:,1].max()+1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = model2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(); plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for i, name in enumerate(load_iris().target_names):
        idx = np.where(y_test==i)
        plt.scatter(X2_test[idx,0], X2_test[idx,1], edgecolors='k', label=name)
    plt.xlabel('Petala L (cm)'); plt.ylabel('Petala W (cm)'); plt.legend()
    buf2 = io.BytesIO(); plt.savefig(buf2, format='png'); buf2.seek(0); plt.close()
    ds_img = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return jsonify({
        "test_metrics": metrics_test,
        "train_metrics": metrics_train,
        "confusion_matrix": cm_img,
        "decision_surface": ds_img
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Modelo não treinado"}), 400
    data = request.get_json()
    try:
        vals = [float(data[key]) for key in ["sepal_length","sepal_width","petal_length","petal_width"]]
    except:
        return jsonify({"error": "Dados inválidos"}), 400
    pred = model.predict([vals])[0]
    name = load_iris().target_names[pred]
    acc = round(model.score(X_train, y_train)*100, 0)
    return jsonify({"predicao": f"{name} (ACC: {acc}%)"})

if __name__ == '__main__':
    app.run(debug=True)