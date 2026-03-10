import os
import streamlit as st

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered",
)

st.title("📧 Email Spam Classifier")
st.markdown("Classify emails or SMS messages as **Spam** or **Ham** using ML.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        **Model:** Best of Naive Bayes, Logistic Regression, SVM, Random Forest

        **Features:** TF-IDF (unigrams + bigrams, 5k features)

        **Dataset:** SMS Spam Collection (UCI)
        """
    )
    st.markdown("---")
    if st.button("🔄 Train Model"):
        with st.spinner("Training... this may take a minute."):
            try:
                from src.pipeline.train_pipeline import TrainPipeline
                pipeline = TrainPipeline()
                best_name, best_score, report = pipeline.run()
                st.success(f"Done! Best model: **{best_name}** (F1={best_score:.4f})")
                st.session_state["report"] = report
            except Exception as e:
                st.error(f"Training failed: {e}")

    if "report" in st.session_state:
        st.markdown("### Last Training Report")
        import pandas as pd
        df_report = pd.DataFrame(st.session_state["report"]).T.round(4)
        st.dataframe(df_report)

# ── Model loading ─────────────────────────────────────────────────────────────
model_ready = os.path.exists("artifacts/model.pkl") and os.path.exists(
    "artifacts/tfidf_vectorizer.pkl"
)

if not model_ready:
    st.warning("Model not trained yet. Click **Train Model** in the sidebar to get started.")
else:
    @st.cache_resource
    def load_pipeline():
        from src.pipeline.predict_pipeline import PredictPipeline
        return PredictPipeline()

    predictor = load_pipeline()

    tab1, tab2 = st.tabs(["Single Message", "Batch Classify"])

    with tab1:
        st.subheader("Classify a message")
        user_input = st.text_area(
            "Paste your email / SMS text here:",
            height=180,
            placeholder="e.g. Congratulations! You've won a FREE iPhone. Click here to claim now!",
        )
        if st.button("Classify", type="primary"):
            if user_input.strip():
                label = predictor.predict([user_input])[0]
                try:
                    prob = predictor.predict_proba([user_input])[0]
                    confidence = prob if label == "spam" else 1 - prob
                except Exception:
                    confidence = None
                conf_str = f"  (confidence: {confidence:.1%})" if confidence else ""
                if label == "spam":
                    st.error(f"🚨 **SPAM**{conf_str}")
                else:
                    st.success(f"✅ **HAM** (not spam){conf_str}")
            else:
                st.warning("Please enter some text.")

    with tab2:
        st.subheader("Classify multiple messages")
        st.markdown("Enter one message per line:")
        batch_input = st.text_area("Messages (one per line):", height=200)
        if st.button("Classify All", type="primary"):
            lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
            if lines:
                import pandas as pd
                labels = predictor.predict(lines)
                results = pd.DataFrame({"Message": lines, "Prediction": labels})
                results["Prediction"] = results["Prediction"].map(
                    {"spam": "🚨 Spam", "ham": "✅ Ham"}
                )
                st.dataframe(results, use_container_width=True)
                spam_count = sum(1 for l in labels if l == "spam")
                st.info(
                    f"Found **{spam_count} spam** and **{len(labels) - spam_count} ham** "
                    f"out of {len(labels)} messages."
                )
            else:
                st.warning("Please enter at least one message.")

    with st.expander("Try example messages"):
        examples = {
            "Spam 1": "WINNER!! You have been selected to receive a £900 prize reward! Call now to claim!",
            "Spam 2": "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 now!",
            "Ham 1": "Hey, are you coming to the meeting tomorrow at 3pm?",
            "Ham 2": "I'll be home by 7. Can you pick up some milk on the way?",
        }
        for ex_label, msg in examples.items():
            col1, col2 = st.columns([4, 1])
            col1.caption(f"[{ex_label}] {msg[:90]}")
            if col2.button("Try", key=ex_label):
                result = predictor.predict([msg])[0]
                if result == "spam":
                    st.error(f"🚨 SPAM — _{msg[:60]}..._")
                else:
                    st.success(f"✅ HAM — _{msg[:60]}..._")
