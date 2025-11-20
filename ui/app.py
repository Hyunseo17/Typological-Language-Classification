import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config.settings import (  # noqa: E402
    ARTIFACTS_DIR,
    DEFAULT_MODEL_PATH,
    LANGUAGE_METADATA,
    TARGET_LANGUAGES,
    TrainingConfig,
)
from src.eval.reporting import save_confusion_plot  # noqa: E402
from src.models.inference import load_pipeline, predict_language  # noqa: E402
from src.models.training_pipeline import train_and_eval  # noqa: E402

st.set_page_config(
    page_title="SVM Language Identifier",
    page_icon="🌐",
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

HIDE_MENU = """
<style>
#MainMenu {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}
header .st-emotion-cache-ucpwe8 {display: none;}
[data-testid="stActionButtonIcon"] {display: none !important;}
</style>
"""
st.markdown(HIDE_MENU, unsafe_allow_html=True)
st.title("🌐 Typological Language Identifier")
st.write(
    "Enter a short passage and the Support Vector Machine classifier will attempt "
    "to identify the language based on typological cues."
)
supported_langs = ", ".join(meta["name"] for meta in LANGUAGE_METADATA.values())
st.caption(f"Supported languages: {supported_langs}")


def list_artifacts() -> list[Path]:
    ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
    artifacts = sorted(ARTIFACTS_DIR.glob("*.joblib"))
    if DEFAULT_MODEL_PATH not in artifacts:
        artifacts.insert(0, DEFAULT_MODEL_PATH)
    return artifacts


def load_metrics(path: Path):
    metrics_file = path.with_suffix(".metrics.json")
    if metrics_file.exists():
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    return None


@st.cache_resource(show_spinner=False)
def get_pipeline(model_path: str):
    return load_pipeline(Path(model_path))


artifact_choices = list_artifacts()
artifact_labels = [p.name for p in artifact_choices]
selected_index = 0 if artifact_choices else None

with st.sidebar:
    st.header("Model Controls")
    if not artifact_choices:
        st.warning("No artifacts found. Train a model to get started.")
    selected_model = st.selectbox(
        "Choose model artifact",
        artifact_choices,
        index=selected_index or 0,
        format_func=lambda p: p.name,
    )

    metrics = load_metrics(selected_model)
    with st.expander("Current model metrics", expanded=False):
        if metrics:
            macro = metrics["summary"].get("macro avg", {})
            st.metric("Accuracy", f"{metrics['summary'].get('accuracy', 0)*100:.2f}%")
            st.metric("Macro F1", f"{macro.get('f1-score', 0)*100:.2f}%")
            st.write("Top-k accuracy:", metrics["top_k"])
        else:
            st.caption("Train the model to generate metrics.")

    with st.expander("Train a model", expanded=False):
        st.write("Upload a CSV with `text` and `language` columns.")
        upload = st.file_uploader("Dataset CSV", type=["csv"])
        artifact_name = st.text_input("Artifact filename", "custom_language_svm.joblib")
        grid = st.checkbox("Enable hyperparameter search", value=False)
        if st.button("Train from CSV", disabled=upload is None):
            try:
                df = pd.read_csv(upload)
                if not {"text", "language"}.issubset(df.columns):
                    st.error("CSV must include `text` and `language` columns.")
                else:
                    config = TrainingConfig()
                    pipeline, metrics_obj, confusion = train_and_eval(
                        df,
                        config,
                        grid_search=grid,
                        n_jobs=-1 if grid else 1,
                    )
                    output_path = ARTIFACTS_DIR / artifact_name
                    joblib.dump(pipeline, output_path)
                    metrics_path = output_path.with_suffix(".metrics.json")
                    metrics_path.write_text(json.dumps(metrics_obj, indent=2), encoding="utf-8")
                    conf_path = output_path.with_suffix(".confusion.png")
                    save_confusion_plot(
                        confusion["confusion"],
                        labels=sorted(df["language"].unique()),
                        path=conf_path,
                    )
                    st.success(f"Trained and saved to {output_path.name}")
                    st.experimental_rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")

pipeline = None
if artifact_choices:
    try:
        pipeline = get_pipeline(str(selected_model))
    except FileNotFoundError:
        pipeline = None

if pipeline is None:
    st.warning(
        "No trained model artifact found. Use the sidebar to train on the sample dataset "
        "or upload your own CSV."
    )
else:
    cols = st.columns([2, 1])
    with cols[0]:
        text = st.text_area(
            "Input text",
            height=220,
            placeholder="Type or paste text here...",
        )
        detect = st.button("Detect Language", type="primary")
    if detect and text.strip():
        with st.spinner("Predicting..."):
            result = predict_language(text, pipeline=pipeline)
        st.success(f"Predicted language: **{result['top_language']}**")
        metadata = LANGUAGE_METADATA.get(result["top_language"], {})
        if metadata:
            st.info(
                f"{metadata.get('name')}\n\n"
                f"_Sample phrase:_ {metadata.get('sample_text')}"
            )
        ranked_df = pd.DataFrame(result["ranked"])
        ranked_df["probability_pct"] = (ranked_df["probability"] * 100).round(2)
        st.subheader("Top predictions")
        st.bar_chart(ranked_df.set_index("language")["probability"])
        st.dataframe(
            ranked_df[["language", "probability_pct", "score"]].head(5),
            use_container_width=True,
        )

    with cols[1]:
        st.caption("Need inspiration?")
        language_cards = list(LANGUAGE_METADATA.values())
        tabs = st.tabs([meta["code"] for meta in language_cards])
        for tab, meta in zip(tabs, language_cards):
            with tab:
                st.write(meta["sample_text"])



