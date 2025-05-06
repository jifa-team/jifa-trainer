
# 🧠 Jifa Trainer — KNN + Estrutura Modular com Flask

> Feito é melhor que o perfeito. — *George Patton* ✨

---

## 🚀 No Princípio...

Antes de tudo, relaxa! 😌  
Este projeto está hospedado em **servidores 0800**, então **aguenta firme entre 30 e 59 segundos** para ele acordar! 😴⚡

👉 [Acesse aqui a versão online bonitinha](https://jifa-trainer.onrender.com/)

---

## 📦 Resumo das Mudanças

Criamos uma estrutura 🏗️ pensada para organizar os modelos e facilitar a vida:

```
models/
├── __init__.py
├── base_model.py
└── knn_model.py
```

### 📚 `base_model.py`

- Classe base abstrata `BaseModel`
- Métodos comuns para:
  - Avaliação dos modelos
  - Geração da matriz de confusão
  - Visualização da superfície de decisão
- Métodos abstratos para obrigar implementação correta nos modelos filhos

### 🔍 `knn_model.py`

- Implementa o modelo **KNN**
- Herda de `BaseModel`
- Implementa os métodos `train` e `predict`
- Adiciona lógica específica do KNN

### 💻 `back.py`

- Agora usa a nova estrutura de modelos
- Código bem mais limpo, modular e expansível

---

## 🌟 Benefícios dessa nova estrutura

✅ **Modularidade** – Cada modelo no seu quadrado (arquivo)  
✅ **Extensibilidade** – Fica fácil adicionar novos modelos  
✅ **Manutenibilidade** – Tudo mais organizado e claro  
✅ **Reutilização** – Métodos compartilhados na classe base  
✅ **Consistência** – Interface padrão para todos os modelos

---

## 🧪 Adicionando um Novo Modelo

1. Criar um novo arquivo em `models/` (ex: `svm_model.py`)
2. Herdar da `BaseModel`
3. Implementar os métodos `train` e `predict`
4. Registrar o modelo no `__init__.py`
5. Adicionar a lógica no `back.py`

---

## 🧠 O que é esse projeto mesmo?

Este é um projeto da disciplina de **Aprendizagem de Máquina** do Professor Dr. **Luís Fabrício de Freitas Souza** na **Universidade Federal do Cariri – UFCA**.  

O objetivo é aplicar de forma prática os conceitos aprendidos sobre algoritmos de aprendizado de máquina 🤓

---

## 👥 Discentes Executores

- Gustavo Ferreira Reinaldo  
- Sayonara Arcanjo da Silva  
- Alexandra Silva de Paula  
- Carlos Eduardo de Lima Lira Santana

---

## 📋 O que foi solicitado?

- Clonar o repositório do tutor 👨‍🏫  
- Incluir novos modelos de aprendizagem de máquina 🤖  
- Criar um layout próprio com novas funcionalidades ✨  

---

## 📚 Wiki de Execução — baseada nos commits

| Commit | Descrição | Link |
|--------|-----------|------|
| #sem-numero   | Resumo do Projeto | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki) |
| #1982e43   | Projeto inicial | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki/2-Projeto-inicial) |
| #aec331c   | Separando as coisas | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki/3-Separando-as-coisas) | 
| #7417bf5   | Adicionando uma home no projeto | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki/4-Adicionando-uma-home-no-projeto) |
| #7724838   | Preparando o projeto para aceitar diferentes modelos | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki/5-Preparando-o-projeto-para-aceitar-diferentes-modelos) |
| #8c4234f   | Um novo modelo no projeto: K‐Means | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki/6-Um-novo-modelo-no-projeto:-K-Means) |
| #b21bd00   | O projeto ao vivo na web | [Ver na Wiki](https://github.com/jifa-team/jifa-trainer/wiki/7-O-projeto-ao-vivo-na-web) |

---

## 🛠️ Agora na Sua Máquina

### ✅ Pré-requisitos

- Python 3.x instalado 🐍
- Pip (gerenciador de pacotes do Python)

### 📦 Clonando o projeto

Clone o repositório ou extraia os arquivos do projeto.

Confirme que você tem no diretório pelo menos:

- `back.py`
- `templates/front.html`
- `requirements.txt`

---

### 🔐 Criando o Ambiente Virtual

```bash
python -m venv venv
```

#### Ativar o ambiente virtual:

- No **Windows (PowerShell)**:
  ```bash
  .\venv\Scripts\Activate
  ```

- No **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

---

### 📥 Instalando as dependências

```bash
pip install -r requirements.txt
```

---

### ▶️ Rodando o projeto

Com o ambiente ativado, rode:

```bash
python back.py
```

Acesse no navegador:

```
http://127.0.0.1:5000
```

---

## 🧱 Estrutura do Projeto

```
JIFA-Trainer/
├── models/ # Modelos de aprendizado
│ ├── init.py
│ ├── base_model.py
│ ├── kmeans_model.py
│ └── knn_model.py
├── static/ # Arquivos estáticos (CSS, JS, imagens)
│ ├── logo.png
│ ├── script.js
│ └── style.css
├── templates/ # Templates HTML (Front-end)
│ ├── front.html
│ └── home.html
├── venv/ # Ambiente virtual Python
├── .gitignore # Arquivos/pastas ignorados pelo Git
├── back.py # Backend principal (Flask)
├── readme.md # Este arquivo lindo que você está lendo 😄
└── requirements.txt # Dependências do projeto
```

---

✨ Projeto com carinho e aprendizado 💡
