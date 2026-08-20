# ToxicGuard-RAG 

**Local AI system for Ukrainian toxic comment moderation with RAG-based explanations.**

ToxicGuard-RAG is a local AI application for detecting toxic comments and generating human-readable explanations of moderation decisions.

The system combines:

* multilingual text embeddings;
* a supervised toxicity classifier;
* FAISS vector similarity search;
* Retrieval-Augmented Generation (RAG);
* a local Ollama LLM;
* Streamlit for the user interface.

The entire pipeline can run locally without sending user comments to a cloud AI API.

---

## Machine Learning

### Embeddings

The project uses:

```text
intfloat/multilingual-e5-small
```

through the `sentence-transformers` library.

The model converts text into dense numerical vectors that capture semantic information.

```text
Text
 ↓
Sentence Transformer
 ↓
768-dimensional embedding
```

---

### Toxicity Classifier

The embeddings are used as input features for a supervised binary classifier.

Current model:

```text
Logistic Regression
```

The classifier predicts the probability that a comment belongs to the toxic class.

A calibrated classification threshold of approximately:

```text
0.708
```

is used instead of the default `0.5`.

This allows the decision boundary to be adjusted according to the desired Precision/Recall trade-off.

---

## RAG

The project uses **Retrieval-Augmented Generation (RAG)** to provide explanations for moderation decisions.

The workflow is:

```text
User Comment
     ↓
Embedding
     ↓
FAISS similarity search
     ↓
Top 3 similar comments
     ↓
Prompt
     ↓
Local LLM
     ↓
Explanation
```

Instead of asking the LLM to determine toxicity on its own, the application first performs classification using the trained ML model.

The LLM is primarily used to generate a human-readable explanation based on the original comment and retrieved examples.

---

## Vector Search

The project uses:

```text
FAISS
```

for efficient similarity search over text embeddings.

FAISS stores the embeddings of previously processed comments and allows the system to retrieve semantically similar examples.

For example:

```text
Query:
"Ти повний ідіот"

        ↓

FAISS

        ↓

Similar comments:
1. "Який же ти дурень"
2. "Ти нічого не розумієш, ідіот"
3. "Замовкни, дурень"
```

These retrieved examples are then provided to the LLM as context.

---

## Local LLM

The project uses **Ollama** to run an LLM locally.

Current model:

```text
qwen2.5:7b
```

The LLM generates concise explanations of why a comment may require moderation.

Example:

```text
Input:
"Ти ідіот, забирайся звідси!"

Output:
"Коментар містить образливе звернення
та використання принизливої лексики."
```

Because the model runs locally, the application does not require a cloud LLM API.

---

## Performance

Current classifier results:

| Metric                   |      Value |
| ------------------------ | ---------: |
| Validation Accuracy      |      ~0.81 |
| Test Accuracy            |      ~0.77 |
| Precision                |      ~0.87 |
| Recall                   |      ~0.64 |
| F1-score                 | ~0.74–0.80 |
| Classification Threshold |     ~0.708 |

The exact metrics may depend on the dataset split and experiment configuration.

Detailed experiments, hyperparameter tuning, threshold calibration, and error analysis are available in the training notebooks.

---

## Tech Stack

| Technology                      | Purpose                     |
| ------------------------------- | --------------------------- |
| Python 3.12                     | Core programming language   |
| PyTorch / Sentence Transformers | Text embeddings             |
| Hugging Face                    | Pre-trained embedding model |
| scikit-learn                    | Logistic Regression         |
| FAISS                           | Vector similarity search    |
| Ollama                          | Local LLM inference         |
| Qwen 2.5 7B                     | Explanation generation      |
| Streamlit                       | Web interface               |
| Pandas / NumPy                  | Data processing             |
| Jupyter Notebook                | Experiments and analysis    |
| Joblib                          | Model serialization         |

---

## Project Structure

```text
Toxic-Guard-RAG/
│
├── app.py                         # Streamlit application
│
├── models/                        # Trained model artifacts
│   ├── classifier_model.joblib
│   ├── faiss_index.faiss
│   └── config.json
│
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # Processed data and RAG texts
│       └── rag_texts.pkl
│
├── notebooks/                     # EDA and model development
│
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

> Model artifacts and large files may be excluded from the Git repository because of GitHub file-size limitations.

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/Bogrudik/Toxic-Comment-Classification.git
cd Toxic-Comment-Classification
```

If your repository has been renamed to `Toxic-Guard-RAG`, replace the repository URL accordingly.

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv

.\venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv

source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download model artifacts

Large trained files are not included directly in the Git repository.

The following files are required:

```text
models/
├── classifier_model.joblib
├── faiss_index.faiss
└── config.json
```

If the project provides the artifacts through Google Drive or another storage service, download them and place them into the `models/` directory.

The processed RAG data should be placed in:

```text
data/processed/rag_texts.pkl
```

---

# Ollama Setup

Install Ollama and download the required model.

Run:

```bash
ollama pull qwen2.5:7b
```

Then start the model:

```bash
ollama run qwen2.5:7b
```

Keep Ollama running while using the application.

---

# Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# Docker

Docker support can be used to containerize the application and provide a reproducible runtime environment.

Build the image:

```bash
docker build -t toxicguard-rag .
```

Run the container:

```bash
docker run -p 8501:8501 toxicguard-rag
```

> If Ollama runs outside the container, the container must be configured to communicate with the Ollama host.

---

# Model Development

The model development process includes:

```text
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Train / Validation / Test Split
      ↓
Text Embeddings
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Threshold Calibration
      ↓
FAISS Index Creation
      ↓
RAG Integration
```

Experiments and analysis are documented in the Jupyter notebooks located in:

```text
notebooks/
```

---

# Future Improvements

* Improve toxicity classification performance.
* Build a more robust threshold calibration procedure.
* Use only toxic examples or separate indexes for more precise RAG retrieval.
* Add a two-threshold moderation system:

  * high-confidence toxic → automatic block;
  * uncertain cases → human moderation.
* Add automated evaluation of LLM-generated explanations.
* Add Docker Compose for running the application and Ollama together.
* Add FastAPI backend for production-style API access.
* Experiment with smaller local LLMs for lower hardware requirements.
* Add monitoring and logging for production deployment.

---

# Project Goals

The project demonstrates practical experience with:

* supervised machine learning;
* NLP;
* text embeddings;
* semantic search;
* vector databases;
* RAG pipelines;
* local LLM inference;
* prompt engineering;
* model evaluation;
* threshold calibration;
* Streamlit;
* Docker.

The main goal is to combine a traditional ML classifier with modern LLM/RAG techniques into a single local AI application.

---

# Author

**Bogdan**

GitHub:
https://github.com/Bogrudik

---

## License

MIT License.
