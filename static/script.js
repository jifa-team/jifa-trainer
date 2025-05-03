// Funções AJAX para treinar, testar e predizer
const btnTrain = document.getElementById('btnTreinar');
const btnTest = document.getElementById('btnTestar');
const loading = document.getElementById('loading');
const content = document.getElementById('resultsContent');
const metricsDiv = document.getElementById('metrics');
const cmImage = document.getElementById('cmImage');
const dsImage = document.getElementById('dsImage');
const alertBox = document.getElementById('resultadoPredict');
const formPredict = document.getElementById('formPredict');

btnTrain.addEventListener('click', async () => {
  btnTrain.disabled = true;
  const res = await fetch('/train', { method: 'POST' });
  const data = await res.json();
  alertBox.textContent = data.message;
  alertBox.className = 'alert success';
  alertBox.style.display = 'block';
  btnTrain.disabled = false;
});

btnTest.addEventListener('click', async () => {
  loading.style.display = 'flex';
  content.style.display = 'none';
  const res = await fetch('/test');
  const data = await res.json();
  // Preencher métricas
  metricsDiv.innerHTML = `
    <h3>Métricas Treino</h3>
    <p>Acurácia: ${data.train_metrics.accuracy * 100}%</p>
    <p>Precisão: ${data.train_metrics.precision * 100}%</p>
    <p>Recall: ${data.train_metrics.recall * 100}%</p>
    <hr>
    <h3>Métricas Teste</h3>
    <p>Acurácia: ${data.test_metrics.accuracy * 100}%</p>
    <p>Precisão: ${data.test_metrics.precision * 100}%</p>
    <p>Recall: ${data.test_metrics.recall * 100}%</p>
  `;
  cmImage.src = `data:image/png;base64,${data.confusion_matrix}`;
  dsImage.src = `data:image/png;base64,${data.decision_surface}`;
  loading.style.display = 'none';
  content.style.display = 'block';
});

formPredict.addEventListener('submit', async e => {
  e.preventDefault();
  const fd = new FormData(formPredict);
  const payload = {};
  for (let [k,v] of fd.entries()) payload[k] = parseFloat(v);
  alertBox.textContent = 'Carregando...';
  alertBox.className = 'alert';
  alertBox.style.display = 'block';
  const res = await fetch('/predict', {
    method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
  });
  const data = await res.json();
  alertBox.textContent = data.predicao || data.error;
  alertBox.className = 'alert success';
});