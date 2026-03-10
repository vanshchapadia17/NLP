import os

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

STATIC_FOLDER = os.path.join(os.path.dirname(__file__), "frontend", "dist")

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="")
CORS(app)

MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "tfidf_vectorizer.pkl")

_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.pipeline.predict_pipeline import PredictPipeline
        _predictor = PredictPipeline()
    return _predictor


def model_ready():
    return os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    full = os.path.join(STATIC_FOLDER, path)
    if path and os.path.exists(full):
        return send_from_directory(STATIC_FOLDER, path)
    return send_from_directory(STATIC_FOLDER, "index.html")


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({"model_ready": model_ready()})


@app.route("/api/predict", methods=["POST"])
def predict():
    if not model_ready():
        return jsonify({"error": "Model not trained yet. Please train first."}), 400

    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    predictor = get_predictor()
    label = predictor.predict([text])[0]

    try:
        prob = predictor.predict_proba([text])[0]
        confidence = prob if label == "spam" else 1 - prob
    except Exception:
        confidence = None

    return jsonify({
        "label": label,
        "confidence": round(confidence, 4) if confidence is not None else None,
    })


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    if not model_ready():
        return jsonify({"error": "Model not trained yet. Please train first."}), 400

    data = request.get_json()
    messages = (data or {}).get("messages", [])
    messages = [m.strip() for m in messages if isinstance(m, str) and m.strip()]
    if not messages:
        return jsonify({"error": "No messages provided."}), 400

    predictor = get_predictor()
    labels = predictor.predict(messages)

    try:
        probs = predictor.predict_proba(messages)
        confidences = [
            round((p if l == "spam" else 1 - p), 4)
            for p, l in zip(probs, labels)
        ]
    except Exception:
        confidences = [None] * len(labels)

    results = [
        {"message": msg, "label": lbl, "confidence": conf}
        for msg, lbl, conf in zip(messages, labels, confidences)
    ]
    spam_count = sum(1 for l in labels if l == "spam")

    return jsonify({
        "results": results,
        "summary": {
            "total": len(results),
            "spam": spam_count,
            "ham": len(results) - spam_count,
        },
    })


@app.route("/api/train", methods=["POST"])
def train():
    try:
        global _predictor
        _predictor = None

        from src.pipeline.train_pipeline import TrainPipeline
        pipeline = TrainPipeline()
        best_name, best_score, report, comparison = pipeline.run()

        def clean_report(r):
            return {
                k: {m: round(float(v), 4) for m, v in metrics.items()}
                for k, metrics in r.items()
            }

        clean_comparison = {
            vectorizer: {
                "best_model": data["best_model"],
                "f1_score": round(float(data["f1_score"]), 4),
                "report": clean_report(data["report"]),
            }
            for vectorizer, data in comparison.items()
        }

        return jsonify({
            "best_model": best_name,
            "f1_score": round(float(best_score), 4),
            "report": clean_report(report),
            "comparison": clean_comparison,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
