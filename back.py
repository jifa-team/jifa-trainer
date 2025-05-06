import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend sem GUI
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from models import KNNModel, KMeansModel

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
        model = KNNModel(n_neighbors=3)
    elif model_type == 'kmeans':
        model = KMeansModel(n_clusters=3)
    else:
        return jsonify({"error": "Modelo não suportado"}), 400
    
    model.train(X_train, y_train)
    model.set_test_data(X_test, y_test)
    return jsonify({"message": "Treinamento concluído"})

@app.route('/test', methods=['GET'])
def test():
    if model is None:
        return jsonify({"error": "Modelo não treinado"}), 400

    # Obtém métricas do modelo
    metrics = model.evaluate()

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
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # Gera matriz de confusão apenas para KNN
    cm_img = None
    if isinstance(model, KNNModel):
        cm_img = gen_confusion_image(model.get_confusion_matrix())

    # Superfície de decisão
    ds_data = model.get_decision_surface()
    plt.figure()
    plt.contourf(ds_data['xx'], ds_data['yy'], ds_data['Z'], alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plota os centros dos clusters para K-Means
    if isinstance(model, KMeansModel):
        plt.scatter(ds_data['centers'][:, 0], ds_data['centers'][:, 1], 
                   c='black', marker='x', s=200, linewidths=3)
    
    for i, name in enumerate(load_iris().target_names):
        idx = np.where(ds_data['y_test']==i)
        plt.scatter(ds_data['X_test'][idx,0], ds_data['X_test'][idx,1], 
                   edgecolors='k', label=name)
    
    plt.xlabel('Petala L (cm)')
    plt.ylabel('Petala W (cm)')
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png')
    buf2.seek(0)
    plt.close()
    ds_img = base64.b64encode(buf2.getvalue()).decode('utf-8')

    return jsonify({
        **metrics,
        "confusion_matrix": cm_img,
        "decision_surface": ds_img,
        "model_type": "knn" if isinstance(model, KNNModel) else "kmeans"
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
    
    if isinstance(model, KNNModel):
        name = load_iris().target_names[pred]
        acc = round(model.evaluate()["test_metrics"]["accuracy"]*100, 0)
        return jsonify({"predicao": f"{name} (ACC: {acc}%)"})
    else:
        # Para K-Means, retorna apenas o cluster
        return jsonify({"predicao": f"Cluster {pred}"})

if __name__ == '__main__':
    #app.run(debug=True) # linha comentada para produção
    app.run()