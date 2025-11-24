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
    # Filter out the sample model artifact
    default_resolved = str(DEFAULT_MODEL_PATH.resolve())
    artifact_paths = [str(a.resolve()) for a in artifacts]
    # Remove sample model from the list if it exists
    artifacts = [a for a in artifacts if str(a.resolve()) != default_resolved]
    return artifacts


def load_metrics(path: Path):
    metrics_file = path.with_suffix(".metrics.json")
    if metrics_file.exists():
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    return None


@st.cache_resource(show_spinner=False)
def get_pipeline(cache_key: str):
    """Load pipeline with cache keyed on a unique identifier per model."""
    # Extract the absolute path from the cache key (format: "filename|abs_path|mtime")
    parts = cache_key.split("|")
    abs_path = parts[1] if len(parts) > 1 else cache_key
    return load_pipeline(Path(abs_path))


artifact_choices = list_artifacts()
artifact_labels = [p.name for p in artifact_choices]
selected_index = 0 if artifact_choices else None

# Track selected model to detect changes and force cache refresh
if "last_selected_model" not in st.session_state:
    st.session_state.last_selected_model = None
if "model_cache_version" not in st.session_state:
    st.session_state.model_cache_version = 0

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
    
    # Track model selection to detect changes and increment cache version
    current_model_key = str(selected_model.resolve()) if selected_model.exists() else None
    if current_model_key != st.session_state.last_selected_model:
        st.session_state.last_selected_model = current_model_key
        st.session_state.model_cache_version += 1

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
                    # Update session state to trigger cache refresh for the newly trained model
                    st.session_state.last_selected_model = None
                    st.session_state.model_cache_version += 1
                    st.experimental_rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Training failed: {exc}")

pipeline = None
if artifact_choices:
    try:
        if not selected_model.exists():
            pipeline = None
        else:
            # Create a truly unique cache key that includes filename, path, modification time, and version
            # This ensures each model gets its own cache entry, even if paths are similar
            abs_model_path = str(selected_model.resolve())
            filename = selected_model.name
            mtime = selected_model.stat().st_mtime
            cache_version = st.session_state.get("model_cache_version", 0)
            # Format: "filename|abs_path|mtime|version" - ensures uniqueness per model file and version
            # The filename ensures different models are distinguished even if paths are similar
            # The mtime ensures retrained models get new cache entries
            # The cache_version ensures cache invalidation when model selection changes
            cache_key = f"{filename}|{abs_model_path}|{mtime:.6f}|{cache_version}"
            pipeline = get_pipeline(cache_key)
    except (FileNotFoundError, OSError) as e:
        st.error(f"Error loading model {selected_model.name}: {e}")
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



