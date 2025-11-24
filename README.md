## Typological Language Identification Project

This repository now focuses on detecting three languages—English, French, and Spanish—using typological cues such as orthography, character n-gram structure, and diacritics. WiLI uses ISO-639-3 labels (`eng`, `fra`, `spa`), so this codebase does the same to stay consistent with the dataset.

### Project Goals
- Build a reproducible dataset pipeline around the WiLI-2018 benchmark and optionally curated subsets.
- Engineer linguistic features suitable for linear Support Vector Machines (SVM) that emphasize character composition, diacritics, and language-specific patterns.
- Train, validate, and benchmark an SVM classifier that scales to a practical subset of languages (5–20 target labels) with balanced class coverage.
- Deliver interpretable evaluation artifacts: accuracy, confusion matrix, per-language precision/recall/F1, and top-k error analyses.

### Data Source
- **WiLI-2018**: Wikipedia Language Identification dataset (235 languages; **exactly 500 samples per language in each of the provided train/test files, i.e. 1 000 total per language**).
- Utilities provide:
  - `python -m src.data.wili_downloader download` to fetch the official archive from <https://zenodo.org/record/841984>.
  - `python -m src.data.wili_downloader prepare-subset --languages eng,fra,spa --limit-per-language -1` to filter to the three target languages and generate new train/val/test splits (70/15/15 by default). Setting `--limit-per-language -1` consumes all available WiLI rows.
  - Derived CSVs plus metadata JSON live in `data/processed/`.

### Feature Engineering
- Character n-gram counts (1–5 grams) with hashing or limited vocabulary to control dimensionality.
- TF-IDF weighting to emphasize discriminative n-grams; includes weighting for special characters (é, ü, ß, ñ, etc.).
- Script-specific tokens (Latin, Cyrillic) captured via Unicode blocks or binary flags.
- Proportions of uppercase, punctuation, whitespace, numerals to capture stylistic cues.
- Optional: subword frequency ratios, vowel/consonant clusters, stop-word presence per language.
- Feature vectorization handled via `scikit-learn` pipeline objects for consistency (`TfidfVectorizer`, `HashingVectorizer`, custom transformers).

### Modeling Approach
- **Classifier**: Linear SVM (`LinearSVC`) for strong margin-based separation; optionally experiment with `SGDClassifier` hinge loss for scalability.
- Hyperparameters tuned via cross-validation (C, class weights, n-gram ranges, max features).
- We'll benchmark against lightweight baselines:
  - Majority class predictor (sanity check).
  - Multinomial Naive Bayes on same features.
- Model artifacts saved with metadata (languages list, vectorizer settings) to support inference scripts.

### Evaluation Metrics
- Overall accuracy on held-out test split.
- Confusion matrix to visualize commonly confused language pairs.
- Macro/micro precision, recall, F1 to capture per-language quality.
- Top-k accuracy (k=3) for UX scenarios where model can propose multiple guesses.
- Error analysis notebooks highlighting misclassified samples and diacritic coverage gaps.

### Project Structure (proposed)
```
├── data/
│   ├── raw/            # WiLI downloads
│   ├── processed/      # Train/val/test splits
├── notebooks/          # Exploratory analysis & prototyping
├── src/
│   ├── config/         # Language lists, vectorizer params
│   ├── data/           # Downloading & preprocessing scripts
│   ├── features/       # Vectorizers & custom feature transformers
│   ├── models/         # Training, hyperparameter search, inference
│   └── eval/           # Metrics, plotting, reporting utilities
├── tests/              # Unit tests for data + feature code
├── README.md           # Project overview (this file)
└── requirements.txt    # Python dependencies (scikit-learn, pandas, etc.)
```

### Current Implementation Snapshot
- `src/config/languages.yaml`: declarative list of the supported languages (currently English, French, Spanish) consumed by data prep + UI (ISO-639-3 codes with optional ISO-639-1 aliases for display).
- `src/data/wili_downloader.py`: downloads WiLI, computes archive checksums, balances language counts, and emits stratified train/val/test CSVs + metadata JSON.
- `src/features/text_vectorizer.py`: char + word TF-IDF builders; `src/features/linguistic_features.py` adds handcrafted ratios (whitespace, diacritics, vowels, etc.).
- `src/models/training_pipeline.py`: Typer CLI with optional grid search; saves sklearn Pipeline artifacts plus JSON metrics + confusion plots.
- `src/models/inference.py`: exposes ranked predictions with normalized probabilities for UI/API consumers.
- `src/eval/metrics.py` & `src/eval/reporting.py`: reusable helpers for top-k accuracy, confusion matrices, and artifact persistence.
- `ui/app.py`: Streamlit UI with model picker, metrics viewer, top-k bar charts, language metadata cards, and an inline CSV-training workflow.
- `notebooks/baseline_svm.ipynb`: quick experiment to validate the enriched TF-IDF + LinearSVC pipeline on the bundled sample data.
- `tests/`: pytest-based smoke tests for feature builders and the inference stack.

### Getting Started
1. Install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Train the sample model (fast, for demo purposes):
   ```
   python -m src.models.training_pipeline sample
   ```
3. Launch the UI:
   ```
   streamlit run ui/app.py
   ```
   Type text in any supported language (EN/FR/ES/DE/IT) and click **Detect Language**.
4. When ready, swap the dataset with WiLI subsets and retrain via
   ```
   python -m src.models.training_pipeline custom data/processed/wili_subset.csv --output-path artifacts/wili_language_svm.joblib --grid-search --jobs 4
   ```
- Prepare a WiLI subset (restricted to English, French, Spanish by default, using all available WiLI examples when `--limit-per-language -1`):
   ```
   python -m src.data.wili_downloader download
   python -m src.data.wili_downloader prepare-subset \
       --languages eng,fra,spa \
       --limit-per-language -1 \
       --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
   ```
   The resulting CSV lives under `data/processed/wili_subset.csv` by default and plugs straight into the training CLI.
- Explore metrics interactively via `notebooks/baseline_svm.ipynb`.
- Inspect artifacts:
   - `artifacts/*.joblib` – sklearn Pipelines
   - `artifacts/*.metrics.json` – accuracy, macro averages, top-k
   - `artifacts/*.confusion.png` – confusion heatmaps
- Run tests:
   ```
   pytest
   ```

### UI Preview
- Sidebar lets you select any artifact in `artifacts/`, view metrics, or upload a CSV to trigger fresh training (with optional grid search).
- Main panel accepts free-form text, renders top-k predictions as bars/table, and surfaces metadata (language name, family, sample phrase).
- Tabs supply ready-to-use sample sentences per language for quick sanity checks.

### Windows Set-up Instructions
**Prerequisites**
- **Python**: Recommended Python `3.11` (Windows). Newer Python versions (e.g., 3.13) may cause binary package conflicts for `numpy`/`scipy`/`scikit-learn` on Windows.
- **Or Conda**: Installing Miniconda/Anaconda is a robust alternative for the scientific stack.
- **PowerShell**: These commands are written for `powershell.exe` (Windows PowerShell 5.1+).

**Quick checklist (recommended)**
- **Install Python 3.11** (or Miniconda)
- **Create a venv** at the project root: `./.venv`
- **Activate the venv** and `pip install -r requirements.txt`
- **Run tests** with `pytest`
- **Train the sample model** and validate inference

**1) Clone repository (if not already)**
```powershell
git clone https://github.com/Hyunseo17/Typological-Language-Classification.git
Set-Location 'Typological-Language-Classification'
```

**2) Option A — Install Python 3.11 via winget (recommended on Windows)**
```powershell
# Run once (requires Windows Package Manager)
winget install --id=Python.Python.3.11 -e --accept-package-agreements --accept-source-agreements
# Verify
py -3.11 --version
```

**3) Option B — Use Miniconda (recommended if you already use conda)**
```powershell
# After installing Miniconda
conda create -n tlc python=3.11 -y
conda activate tlc
# Install heavy scientific packages from conda-forge
conda install -c conda-forge numpy scipy scikit-learn pandas matplotlib seaborn -y
# Then install the remaining requirements via pip
pip install -r .\requirements.txt
```

**4) Create & activate a virtual environment (Python 3.11)**
```powershell
# From project root
py -3.11 -m venv .venv
. .\.venv\Scripts\Activate.ps1
# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel
# Install project dependencies
python -m pip install -r .\requirements.txt
```

Notes:
- If you use system Python 3.13, you may hit build errors for `scikit-learn`/`numpy`/`scipy` on Windows; prefer Python 3.11 or conda as above.

**5) Run tests**
```powershell
# Run full test suite
python -m pytest -q
# Run a single test file
python -m pytest tests/test_features.py -q
# Run a single test function
python -m pytest tests/test_features.py::test_char_vectorizer_builds -q
```

**6) Train sample model (creates artifacts)**
```powershell
# Trains on small bundled sample dataset and writes artifacts/
python -m src.models.training_pipeline sample
```
Outputs:
- `artifacts/sample_language_svm.joblib` — trained model
- `artifacts/*.metrics.json` — metrics JSON
- `artifacts/*.confusion.png` — confusion matrix plot

**7) Quick inference (command line)**
```powershell
python -c "from src.models.inference import load_pipeline, predict_language; p=load_pipeline(); print(predict_language('Hello world', pipeline=p))"
```

**8) Run the Streamlit UI**
```powershell
streamlit run ui/app.py
```

**Troubleshooting**
- Error building `scikit-learn` or `numpy` on Windows: this usually means pip attempted to compile packages from source because no compatible wheel exists for your Python version. Solutions:
  - Use Python 3.11 (recommended) or conda (install prebuilt binaries).
  - Install Microsoft Visual C++ Build Tools, CMake and Fortran toolchains — only for advanced users; building `scipy` on Windows is complex.
- If pytest reports `ModuleNotFoundError: No module named 'src'`, ensure you run pytest from the project root or set `PYTHONPATH`:
```powershell
$env:PYTHONPATH = '.'
python -m pytest -q
```

# Typological-Language-Classification
