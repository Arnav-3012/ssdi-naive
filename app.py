import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import matplotlib.pyplot as plt


# ----------------------------
# UI config
# ----------------------------
st.set_page_config(page_title="ML App: Naive Bayes + Regression", layout="wide")
st.title("ML App: Naive Bayes (Classification) + Linear Regression (Regression)")
st.caption("Upload CSV → choose problem type → pick model → pick target/features → split → evaluate.")


# ----------------------------
# Helpers
# ----------------------------
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def plot_confusion(cm: np.ndarray, labels):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha="right")
    plt.yticks(ticks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    return fig


def plot_reg_scatter(y_true, y_pred):
    fig = plt.figure()
    plt.scatter(y_true, y_pred)
    plt.title("Predicted vs True")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.tight_layout()
    return fig


def make_dense_onehot():
    """
    GaussianNB cannot accept sparse input.
    OneHotEncoder is sparse by default.
    So we force dense output.
    Handles both new/old sklearn versions.
    """
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame, feature_cols, do_impute=True, do_scale=True):
    X = df[feature_cols]
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    num_steps = []
    cat_steps = []

    if do_impute:
        num_steps.append(("imputer", SimpleImputer(strategy="median")))
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

    if do_scale:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(steps=num_steps) if num_steps else "passthrough"

    # IMPORTANT: dense onehot so GaussianNB won't crash
    cat_pipe = Pipeline(steps=cat_steps + [("onehot", make_dense_onehot())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_cols, categorical_cols


def get_classification_targets(df: pd.DataFrame, max_unique: int = 20):
    """
    Classification target candidates:
    - object/category columns
    - bool columns
    - numeric columns with <= max_unique unique values (discrete)
    """
    targets = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        nunique = s.nunique()
        if s.dtype == "object":
            targets.append(col)
        elif pd.api.types.is_bool_dtype(s):
            targets.append(col)
        elif pd.api.types.is_numeric_dtype(s) and nunique <= max_unique:
            targets.append(col)
    return targets


def get_regression_targets(df: pd.DataFrame, min_unique: int = 21):
    """
    Regression target candidates:
    - numeric columns with >= min_unique unique values (continuous-ish)
    """
    targets = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue

        nunique = s.nunique()
        if pd.api.types.is_numeric_dtype(s) and nunique >= min_unique:
            targets.append(col)
    return targets


def min_class_count(y: pd.Series) -> int:
    vc = y.value_counts(dropna=True)
    return int(vc.min()) if len(vc) else 0


# ----------------------------
# Sidebar: Upload
# ----------------------------
st.sidebar.header("1) Upload Dataset (CSV)")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = load_csv(uploaded)
df = df.dropna(axis=1, how="all")  # drop fully empty columns

st.subheader("Dataset Preview")
st.dataframe(df.head(25), use_container_width=True)
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

with st.expander("Quick diagnostics"):
    st.write("**Column types:**")
    st.write(df.dtypes.astype(str))
    st.write("**Unique counts (top 25 cols):**")
    st.write(df.nunique(dropna=True).head(25))
    st.write("**Missing values per column:**")
    st.write(df.isna().sum())


# ----------------------------
# Sidebar: Problem type + MODEL SELECT
# ----------------------------
st.sidebar.header("2) Problem Setup")
problem_type = st.sidebar.selectbox("Problem type", ["Classification", "Regression"])

if problem_type == "Classification":
    model_name = st.sidebar.selectbox("Choose model (classification)", ["Gaussian Naive Bayes (GaussianNB)"])
else:
    model_name = st.sidebar.selectbox("Choose model (regression)", ["Linear Regression (LinearRegression)"])


# ----------------------------
# Sidebar: Target + Features (FILTERED)
# ----------------------------
if df.shape[1] < 2:
    st.error("Dataset must have at least 2 columns.")
    st.stop()

if problem_type == "Classification":
    target_options = get_classification_targets(df, max_unique=20)
else:
    target_options = get_regression_targets(df, min_unique=21)

if not target_options:
    st.error(
        f"No valid target columns detected for **{problem_type}**.\n\n"
        "Classification: target must be discrete (few unique values or text labels).\n"
        "Regression: target must be numeric with many unique values."
    )
    st.stop()

target_col = st.sidebar.selectbox("Target column (y)", options=target_options)

feature_candidates = [c for c in df.columns if c != target_col]
feature_cols = st.sidebar.multiselect(
    "Feature columns (X)",
    options=feature_candidates,
    default=feature_candidates[: min(8, len(feature_candidates))]
)

if not feature_cols:
    st.error("Pick at least 1 feature column.")
    st.stop()


# ----------------------------
# Sidebar: Preprocessing + Split
# ----------------------------
st.sidebar.header("3) Preprocessing")
do_impute = st.sidebar.checkbox("Impute missing values", value=True)
do_scale = st.sidebar.checkbox("Scale numeric features", value=True)

st.sidebar.header("4) Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=100000, value=42, step=1)

stratify_opt = False
if problem_type == "Classification":
    stratify_opt = st.sidebar.checkbox("Stratify (classification only)", value=True)

st.sidebar.header("5) Evaluate")
run = st.sidebar.button("Evaluate ✅", type="primary")

if not run:
    st.info("Configure settings and click **Evaluate**.")
    st.stop()


# ----------------------------
# Prepare X, y
# ----------------------------
if df[target_col].isna().any():
    st.warning("Target has missing values; dropping those rows.")
    df = df.loc[~df[target_col].isna()].copy()

X = df[feature_cols]
y = df[target_col]

# Guardrails
if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
    st.error("Regression requires a numeric target column.")
    st.stop()

if problem_type == "Classification" and y.nunique(dropna=True) < 2:
    st.error("Classification target must have at least 2 classes.")
    st.stop()

# stratify safety
stratify = None
if problem_type == "Classification" and stratify_opt:
    mcc = min_class_count(y)
    if mcc < 2:
        st.warning("Stratify disabled automatically: at least one class has <2 samples.")
        stratify = None
    else:
        stratify = y

# Split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify,
    )
except Exception as e:
    st.warning(f"Split failed with stratify. Retrying without stratify. Reason: {e}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=None,
    )


# ----------------------------
# Preprocess + model
# ----------------------------
preprocessor, numeric_cols, categorical_cols = build_preprocessor(
    df, feature_cols, do_impute=do_impute, do_scale=do_scale
)

if problem_type == "Classification":
    model = GaussianNB()
else:
    model = LinearRegression()

pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

try:
    pipe.fit(X_train, y_train)
except Exception as e:
    st.error(f"Training failed: {e}")
    st.stop()

y_pred = pipe.predict(X_test)


# ----------------------------
# Results
# ----------------------------
st.subheader("Results")

with st.expander("Pipeline details"):
    st.write(f"**Problem type:** {problem_type}")
    st.write(f"**Model selected:** {model_name}")
    st.write(f"**Numeric features:** {numeric_cols}")
    st.write(f"**Categorical features:** {categorical_cols}")
    st.write(f"**Imputation:** {'ON' if do_impute else 'OFF'}")
    st.write(f"**Scaling:** {'ON' if do_scale else 'OFF'}")

if problem_type == "Classification":
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision (weighted)", f"{prec:.4f}")
    c3.metric("Recall (weighted)", f"{rec:.4f}")
    c4.metric("F1 (weighted)", f"{f1:.4f}")

    st.write("### Classification Report")
    st.code(classification_report(y_test, y_pred, zero_division=0), language="text")

    labels = sorted(pd.Series(y_test).unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    st.write("### Confusion Matrix")
    st.pyplot(plot_confusion(cm, labels))

else:
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("R²", f"{r2:.4f}")

    st.write("### Predicted vs True Plot")
    st.pyplot(plot_reg_scatter(y_test, y_pred))

# ----------------------------
# Predictions preview + download
# ----------------------------
st.write("### Predictions Preview")
pred_df = X_test.copy()
pred_df["y_true"] = y_test.values
pred_df["y_pred"] = y_pred
st.dataframe(pred_df.head(40), use_container_width=True)

st.download_button(
    "Download predictions CSV",
    data=pred_df.to_csv(index=False).encode("utf-8"),
    file_name="predictions.csv",
    mime="text/csv",
)