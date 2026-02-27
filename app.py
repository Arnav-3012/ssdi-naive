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
st.set_page_config(page_title="ML App: Descriptive Analysis + NB/LR", layout="wide")
st.title("ML App: Descriptive Analysis + GaussianNB (Classification) / LinearRegression (Regression)")
st.caption("Upload CSV → Descriptive Analysis (EDA) → choose problem type → select target/features → train/test split → evaluate.")


# ----------------------------
# Helpers
# ----------------------------
def load_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def make_dense_onehot():
    """GaussianNB cannot accept sparse input; OneHotEncoder is sparse by default."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)  # sklearn < 1.2


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
# Plot helpers (NO hist, NO box, NO heatmap)
# ----------------------------
def plot_barh_series(s: pd.Series, title: str, xlabel: str):
    fig = plt.figure()
    s = s.sort_values()
    s.plot(kind="barh")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    return fig


def plot_bar_series(s: pd.Series, title: str, ylabel: str):
    fig = plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


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


# ----------------------------
# Descriptive Analysis (Full Statistical Summary)
# ----------------------------
def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["number", "bool"])
    if num.shape[1] == 0:
        return pd.DataFrame()

    out = num.describe().T  # count, mean, std, min, 25, 50, 75, max
    out["missing"] = df[num.columns].isna().sum().values
    out["missing_%"] = (out["missing"] / len(df) * 100).round(2)
    out["unique"] = df[num.columns].nunique(dropna=True).values

    # extra stats
    # avoid crash on bool columns for skew/kurt
    safe_num = num.copy()
    for c in safe_num.columns:
        if pd.api.types.is_bool_dtype(safe_num[c]):
            safe_num[c] = safe_num[c].astype(int)

    out["skew"] = safe_num.skew(numeric_only=True).values
    out["kurtosis"] = safe_num.kurtosis(numeric_only=True).values

    # coefficient of variation = std/mean
    out["cv"] = (out["std"] / out["mean"].replace(0, np.nan)).values
    out["cv"] = out["cv"].replace([np.inf, -np.inf], np.nan)

    return out.reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat = df.select_dtypes(include=["object", "category"])
    if cat.shape[1] == 0:
        return pd.DataFrame()

    desc = cat.describe().T  # count, unique, top, freq
    desc["missing"] = df[cat.columns].isna().sum().values
    desc["missing_%"] = (desc["missing"] / len(df) * 100).round(2)
    desc = desc.reset_index().rename(columns={"index": "column"})
    return desc


def top_missing_series(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(top_n)
    return missing


def dtype_distribution(df: pd.DataFrame) -> pd.Series:
    return df.dtypes.astype(str).value_counts()


def top_unique_series(df: pd.DataFrame, top_n: int = 15) -> pd.Series:
    u = df.nunique(dropna=True).sort_values(ascending=False).head(top_n)
    return u


def top_cv_series(num_summary: pd.DataFrame, top_n: int = 10) -> pd.Series:
    if num_summary.empty:
        return pd.Series(dtype=float)
    tmp = num_summary[["column", "cv"]].dropna().copy()
    tmp = tmp.sort_values("cv", ascending=False).head(top_n)
    return pd.Series(tmp["cv"].values, index=tmp["column"].values)


def top_corr_with_target(df: pd.DataFrame, target_col: str, top_n: int = 10) -> pd.Series:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col not in num_cols:
        return pd.Series(dtype=float)
    cols = [c for c in num_cols if c != target_col]
    if not cols:
        return pd.Series(dtype=float)

    corr = df[cols + [target_col]].corr(numeric_only=True)[target_col].drop(target_col)
    corr = corr.reindex(corr.abs().sort_values(ascending=False).head(top_n).index)
    return corr


# ----------------------------
# Sidebar: Upload
# ----------------------------
st.sidebar.header("1) Upload Dataset (CSV)")
uploaded = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

df = load_csv(uploaded)
df = df.dropna(axis=1, how="all")

st.subheader("Dataset Preview")
st.dataframe(df.head(25), use_container_width=True)
st.caption(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")


# ----------------------------
# Sidebar: Problem type + model (dropdown restored)
# ----------------------------
st.sidebar.header("2) Problem Setup")
problem_type = st.sidebar.selectbox("Problem type", ["Classification", "Regression"])

if problem_type == "Classification":
    model_name = st.sidebar.selectbox(
        "Choose model (classification)",
        ["Gaussian Naive Bayes (GaussianNB)"]
    )
else:
    model_name = st.sidebar.selectbox(
        "Choose model (regression)",
        ["Linear Regression (LinearRegression)"]
    )

# ----------------------------
# Sidebar: Target + Features (filtered)
# ----------------------------
if df.shape[1] < 2:
    st.error("Dataset must have at least 2 columns.")
    st.stop()

target_options = get_classification_targets(df, 20) if problem_type == "Classification" else get_regression_targets(df, 21)

if not target_options:
    st.error(
        f"No valid target columns detected for **{problem_type}**.\n\n"
        "Classification: target must be discrete (few unique values or labels).\n"
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


# ============================================================
# ✅ NEW SECTION: Descriptive Analysis (FULL STATS + MULTI-GRAPH)
# ============================================================
st.markdown("## 📌 Descriptive Analysis (Full Statistical Summary + Multi-Graph EDA)")

tab_over, tab_stats, tab_target = st.tabs(
    ["Multi-Graph Overview", "Full Statistical Summary", "Target Analysis"]
)

with tab_over:
    left, right = st.columns(2)

    with left:
        st.write("### Column Data Types")
        dt = dtype_distribution(df)
        st.pyplot(plot_bar_series(dt, "Column Data Types Distribution", "num columns"))

    with right:
        st.write("### Missing Values (Top Columns)")
        ms = top_missing_series(df, top_n=15)
        if len(ms) > 0:
            st.pyplot(plot_barh_series(ms, "Missing Values (Top 15)", "missing count"))
        else:
            st.success("No missing values ✅")

    left2, right2 = st.columns(2)

    with left2:
        st.write("### Unique Values (Top Columns)")
        uq = top_unique_series(df, top_n=15)
        st.pyplot(plot_barh_series(uq, "Unique Values Count (Top 15)", "unique count"))

    with right2:
        st.write("### Numeric Spread (Top CV Columns)")
        num_sum = numeric_summary(df)
        cv = top_cv_series(num_sum, top_n=10)
        if len(cv) > 0:
            st.pyplot(plot_bar_series(cv, "Highest Variability Features (Top 10 CV)", "CV = std/mean"))
            st.caption("Higher CV → feature varies a lot compared to its mean.")
        else:
            st.info("No numeric columns with valid CV found.")

with tab_stats:
    st.write("### Numeric Summary (Full)")
    num_sum = numeric_summary(df)
    if not num_sum.empty:
        st.dataframe(num_sum, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    st.write("### Categorical Summary (Full)")
    cat_sum = categorical_summary(df)
    if not cat_sum.empty:
        st.dataframe(cat_sum, use_container_width=True)
    else:
        st.info("No categorical columns found.")

with tab_target:
    st.write("### Target Distribution / Summary")

    if problem_type == "Classification":
        y_tmp = df[target_col].dropna()
        vc = y_tmp.value_counts(dropna=True).head(20)
        st.pyplot(plot_bar_series(vc, f"Target Class Counts (Top 20): {target_col}", "count"))
        st.caption("If you see too many unique classes (almost every row different), it’s NOT a good classification target.")

    else:
        # Regression target summary without histogram
        if not pd.api.types.is_numeric_dtype(df[target_col]):
            st.error("Regression target must be numeric.")
        else:
            y_tmp = df[target_col].dropna()

            five = pd.Series({
                "min": float(y_tmp.min()),
                "Q1": float(y_tmp.quantile(0.25)),
                "median": float(y_tmp.median()),
                "Q3": float(y_tmp.quantile(0.75)),
                "max": float(y_tmp.max()),
            })
            st.pyplot(plot_bar_series(five, f"Target 5-number Summary: {target_col}", "value"))

            basic = pd.Series({
                "mean": float(y_tmp.mean()),
                "std": float(y_tmp.std()),
            })
            st.pyplot(plot_bar_series(basic, f"Target Mean & Std: {target_col}", "value"))

            st.write("### Top Correlations with Target (Numeric features)")
            corr = top_corr_with_target(df, target_col, top_n=10)
            if len(corr) > 0:
                st.pyplot(plot_bar_series(corr, "Top Correlations with Target (Top 10)", "corr"))
            else:
                st.info("Not enough numeric features (or target not numeric) to compute correlation.")


# ----------------------------
# Sidebar: Preprocessing + split
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
    st.info("Set options and click **Evaluate** to train the model.")
    st.stop()


# ----------------------------
# Prepare X, y
# ----------------------------
if df[target_col].isna().any():
    st.warning("Target has missing values; dropping those rows.")
    df = df.loc[~df[target_col].isna()].copy()

X = df[feature_cols]
y = df[target_col]

if problem_type == "Regression" and not pd.api.types.is_numeric_dtype(y):
    st.error("Regression requires a numeric target column.")
    st.stop()

if problem_type == "Classification" and y.nunique(dropna=True) < 2:
    st.error("Classification target must have at least 2 classes.")
    st.stop()

stratify = None
if problem_type == "Classification" and stratify_opt:
    if min_class_count(y) < 2:
        st.warning("Stratify disabled automatically: at least one class has <2 samples.")
        stratify = None
    else:
        stratify = y

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
# Model pipeline
# ----------------------------
preprocessor, numeric_cols, categorical_cols = build_preprocessor(
    df, feature_cols, do_impute=do_impute, do_scale=do_scale
)

model = GaussianNB() if problem_type == "Classification" else LinearRegression()
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
st.markdown("## ✅ Model Results")

with st.expander("Pipeline details"):
    st.write(f"**Problem type:** {problem_type}")
    st.write(f"**Model:** {model_name}")
    st.write(f"**Numeric features:** {numeric_cols}")
    st.write(f"**Categorical features:** {categorical_cols}")
    st.write(f"**Imputation:** {'ON' if do_impute else 'OFF'}")
    st.write(f"**Scaling:** {'ON' if do_scale else 'OFF'}")

if problem_type == "Classification":
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision (weighted)", f"{prec:.4f}")
    m3.metric("Recall (weighted)", f"{rec:.4f}")
    m4.metric("F1 (weighted)", f"{f1:.4f}")

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

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{mae:.4f}")
    m2.metric("RMSE", f"{rmse:.4f}")
    m3.metric("R²", f"{r2:.4f}")

    st.write("### Predicted vs True")
    st.pyplot(plot_reg_scatter(y_test, y_pred))

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