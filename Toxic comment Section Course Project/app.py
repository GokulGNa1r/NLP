"""
Toxic Comment Detector: web app (Streamlit)

Run:
    python -m streamlit run app.py
Keep this file in the same folder as toxic_detector.py and small_toxic_dataset.csv.
"""

import joblib
import streamlit as st

from toxic_detector import DEFAULT_DATA, MODEL_PATH, load_csv, predict, train_and_evaluate

st.set_page_config(page_title="Toxic Comment Detector", page_icon="🛡️", layout="centered")

LABEL_NAMES = {
    "toxic": "Toxic",
    "severe_toxic": "Severe toxic",
    "obscene": "Obscene",
    "threat": "Threat",
    "insult": "Insult",
    "identity_hate": "Identity hate",
}
EXAMPLES = {
    "Polite": "Thanks, this was really helpful!",
    "Insult": "You are an idiot and nobody wants you here.",
    "Threat": "I will find you and hurt you.",
}


@st.cache_resource(show_spinner=False)
def get_bundle():
    """Load the saved model; train it first if it does not exist yet."""
    if not MODEL_PATH.exists():
        train_and_evaluate(load_csv(DEFAULT_DATA))
    return joblib.load(MODEL_PATH)


def set_example(text):
    st.session_state["comment"] = text


st.title("🛡️ Toxic Comment Detector")
st.caption("NLP course project · TF-IDF + Logistic Regression · six toxicity labels")

with st.spinner("Loading model (first run trains it, this takes a few seconds)..."):
    bundle = get_bundle()

st.write("Try an example or type your own comment:")
cols = st.columns(len(EXAMPLES))
for col, (name, text) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(text,), use_container_width=True)

comment = st.text_area("Comment", key="comment", height=120, placeholder="Type a comment here...")
threshold = st.slider("Flag a label when its score is at least", 0.1, 0.9, 0.5, 0.05)

if st.button("Analyze", type="primary"):
    if not comment.strip():
        st.warning("Please type a comment first.")
    else:
        scores, flagged = predict(comment, bundle, threshold)
        if flagged:
            st.error("⚠️ Toxic content detected: " + ", ".join(LABEL_NAMES[f] for f in flagged))
        else:
            st.success("✅ This comment looks clean.")
        st.subheader("Scores")
        for key, value in scores.items():
            st.progress(float(value), text=f"{LABEL_NAMES[key]}: {value:.2f}")

st.divider()
st.caption("Trained on a small sample dataset, so treat results as a demo, not a moderation tool.")
