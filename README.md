# Email / SMS Spam Classifier

A full-stack NLP application that classifies messages as **spam** or **ham** (not spam). It features a Flask REST API backend, a React + Vite frontend, and a modular ML pipeline with multiple classifiers.

---

## Features

- Single and batch message classification
- Confidence score for each prediction
- Train/retrain model via the UI or API
- TF-IDF vectorization with text preprocessing (lowercasing, stopword removal, stemming)
- Automatic best-model selection from: Naive Bayes, Logistic Regression, Linear SVC, Random Forest
- Evaluation report (Accuracy, Precision, Recall, F1) after each training run

---

## Project Structure

```
NLP/
├── app.py                  # Flask API server
├── requirements.txt        # Python dependencies
├── setup.py
├── artifacts/              # Saved model and vectorizer (.pkl)
├── data/                   # Raw dataset (SMSSpamCollection, CSVs)
├── notebooks/              # Jupyter exploration notebooks
├── logs/                   # Training logs
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py  # Text cleaning + TF-IDF
│   │   └── model_trainer.py        # Multi-model training & selection
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
└── frontend/               # React + Vite UI
    ├── src/
    │   ├── components/
    │   │   ├── SingleClassify.jsx  # Single message classifier
    │   │   ├── BatchClassify.jsx   # Batch message classifier
    │   │   └── TrainModel.jsx      # Trigger training from UI
    │   └── App.jsx
    └── package.json
```

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- Node.js 18+ (for frontend)

### 1. Clone the repository

```bash
git clone <repo-url>
cd NLP
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Build the frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

### 4. Run the Flask server

```bash
python app.py
```

The app will be available at `http://localhost:5000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Check if model is trained and ready |
| POST | `/api/train` | Train the model on the dataset |
| POST | `/api/predict` | Classify a single message |
| POST | `/api/predict-batch` | Classify multiple messages |

### Example: Single Prediction

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free prize. Click here now."}'
```

**Response:**
```json
{
  "label": "spam",
  "confidence": 0.9871
}
```

### Example: Batch Prediction

```bash
curl -X POST http://localhost:5000/api/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"messages": ["Free prize winner!", "Meeting at 3pm tomorrow"]}'
```

**Response:**
```json
{
  "results": [
    {"message": "Free prize winner!", "label": "spam", "confidence": 0.9512},
    {"message": "Meeting at 3pm tomorrow", "label": "ham", "confidence": 0.9743}
  ],
  "summary": {"total": 2, "spam": 1, "ham": 1}
}
```

---

## ML Pipeline

### Text Preprocessing

- Lowercase conversion
- URL, email, and phone number removal
- Punctuation and digit stripping
- Stopword removal (NLTK)
- Porter Stemming (NLTK)

### Vectorization

- TF-IDF with unigrams and bigrams (`ngram_range=(1,2)`)
- Top 5000 features, `sublinear_tf=True`

### Models Evaluated

| Model | Notes |
|-------|-------|
| Multinomial Naive Bayes | Fast baseline |
| Logistic Regression | Strong general performer |
| Linear SVC | Effective for high-dimensional text |
| Random Forest | Ensemble method |

The best model by **F1 score** is automatically selected and saved to `artifacts/model.pkl`.

---

## Dataset

The model is trained on the [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection), containing 5,574 labeled SMS messages (spam/ham).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask, Flask-CORS |
| ML | scikit-learn, NLTK, NumPy, pandas |
| Frontend | React 18, Vite |
| Serialization | pickle |
