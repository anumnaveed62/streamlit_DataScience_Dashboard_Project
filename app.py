import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Data Science Project Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECTS = {
    "01 Stock Price": "01_stock_price",
    "02 DNA Count": "02_dna_count",
    "03 Basketball EDA": "03_basketball_eda",
    "04 Football EDA": "04_football_eda",
    "05 S&P 500 EDA": "05_sp500_eda",
    "06 Crypto EDA": "06_crypto_eda",
    "07 Iris Classification": "07_iris_classification",
    "08 Penguins Classification": "08_penguins_classification",
    "09 Boston Regression": "09_boston_regression",
    "10 Solubility Regression": "10_solubility_regression",
    "11 Heroku Deploy": "11_heroku_deploy",
    "12 Streamlit Sharing": "12_streamlit_sharing",
}

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 8px;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)


def find_csv(folder):
    if not os.path.exists(folder):
        return []
    return glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)


def load_first_csv(folder):
    files = find_csv(folder)
    if not files:
        return None, None
    for file in files:
        try:
            return pd.read_csv(file), file
        except Exception:
            continue
    return None, files[0]


def dataset_info(df):
    st.subheader("Dataset")
    st.dataframe(df, use_container_width=True, height=400)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Duplicate Rows", int(df.duplicated().sum()))

    with st.expander("Column Information"):
        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": [str(x) for x in df.dtypes],
            "Missing": df.isnull().sum().values,
            "Unique": df.nunique().values,
        })
        st.dataframe(info, use_container_width=True)


def generic_eda(df):
    dataset_info(df)

    st.subheader("Descriptive Statistics")
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True
    )

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    if numeric:
        st.subheader("Visualization")

        col = st.selectbox("Select numeric column", numeric)

        chart_type = st.selectbox(
            "Chart type",
            ["Histogram", "Line Chart", "Box Plot"]
        )

        fig, ax = plt.subplots()

        if chart_type == "Histogram":
            ax.hist(df[col].dropna(), bins=30)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        elif chart_type == "Line Chart":
            ax.plot(df[col].dropna())
            ax.set_xlabel("Index")
            ax.set_ylabel(col)

        else:
            ax.boxplot(df[col].dropna())
            ax.set_ylabel(col)

        ax.set_title(f"{chart_type}: {col}")
        st.pyplot(fig)


def upload_or_load(folder, key):
    df, path = load_first_csv(folder)

    uploaded = st.file_uploader(
        "Upload CSV dataset (optional)",
        type=["csv"],
        key=key
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        path = uploaded.name

    if df is not None:
        st.caption(f"Dataset source: {path}")

    return df


def home_page():
    st.markdown(
        '<div class="main-title">📊 Data Science Project Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'One Streamlit dashboard for all 12 projects'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Use the sidebar to select any project. "
        "CSV files inside the project folders are detected automatically."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Projects", 12)
    c2.metric("EDA Projects", 5)
    c3.metric("ML Projects", 4)
    c4.metric("Deployment Projects", 2)

    st.subheader("Project Categories")

    st.markdown("""
    **Exploratory Data Analysis**
    - Stock Price
    - Basketball
    - Football
    - S&P 500
    - Crypto

    **Machine Learning**
    - Iris Classification
    - Penguins Classification
    - Boston Regression
    - Solubility Regression

    **Other**
    - DNA Count
    - Heroku Deployment
    - Streamlit Sharing
    """)


def stock_price_page():
    st.header("📈 Stock Price")

    df = upload_or_load(
        PROJECTS["01 Stock Price"],
        "stock_upload"
    )

    if df is None:
        st.warning("No stock CSV was found. Upload one above.")
        return

    dataset_info(df)

    numeric = df.select_dtypes(include=np.number).columns.tolist()

    if numeric:
        col = st.selectbox(
            "Price column",
            numeric,
            key="stock_column"
        )

        fig, ax = plt.subplots()
        ax.plot(df[col].dropna())
        ax.set_title("Stock Price Trend")
        ax.set_xlabel("Time / Row")
        ax.set_ylabel(col)
        st.pyplot(fig)


def dna_page():
    st.header("🧬 DNA Count")

    sequence = st.text_area(
        "Enter DNA sequence",
        "ATGCGATCGATCGATCGATCG"
    )

    sequence = sequence.upper().replace(" ", "").replace("\n", "")

    invalid = sorted(set(sequence) - {"A", "T", "G", "C"})

    if invalid:
        st.error(f"Invalid bases: {', '.join(invalid)}")
        return

    if not sequence:
        st.warning("Enter a DNA sequence.")
        return

    counts = {base: sequence.count(base) for base in "ATGC"}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("A", counts["A"])
    c2.metric("T", counts["T"])
    c3.metric("G", counts["G"])
    c4.metric("C", counts["C"])

    gc = (counts["G"] + counts["C"]) / len(sequence) * 100

    st.metric("Total Bases", len(sequence))
    st.metric("GC Content", f"{gc:.2f}%")

    fig, ax = plt.subplots()
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title("Nucleotide Count")
    ax.set_xlabel("Base")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def eda_page(project_key, title):
    st.header(title)

    df = upload_or_load(
        PROJECTS[project_key],
        f"{project_key}_upload"
    )

    if df is None:
        st.warning("No CSV was found. Upload a dataset above.")
        return

    generic_eda(df)


def iris_page():
    st.header("🌸 Iris Classification")

    iris = load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.metric("Test Accuracy", f"{accuracy:.2%}")

    st.subheader("Enter Flower Measurements")

    c1, c2 = st.columns(2)

    with c1:
        sepal_length = st.number_input(
            "Sepal Length",
            min_value=0.0,
            value=5.1
        )
        sepal_width = st.number_input(
            "Sepal Width",
            min_value=0.0,
            value=3.5
        )

    with c2:
        petal_length = st.number_input(
            "Petal Length",
            min_value=0.0,
            value=1.4
        )
        petal_width = st.number_input(
            "Petal Width",
            min_value=0.0,
            value=0.2
        )

    if st.button("Predict Iris"):
        sample = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        pred = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]

        st.success(
            f"Prediction: {iris.target_names[pred]}"
        )

        st.write("Class probabilities:")
        st.dataframe(
            pd.DataFrame({
                "Class": iris.target_names,
                "Probability": probability
            }),
            use_container_width=True
        )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=iris.target_names,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).T,
        use_container_width=True
    )


def penguins_page():
    st.header("🐧 Penguins Classification")

    df = upload_or_load(
        PROJECTS["08 Penguins Classification"],
        "penguins_upload"
    )

    if df is None:
        st.info("Upload your penguins CSV dataset.")
        return

    dataset_info(df)

    target = st.selectbox(
        "Target column",
        df.columns
    )

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    features = [c for c in numeric if c != target]

    if not features:
        st.error("No numeric feature columns available.")
        return

    selected = st.multiselect(
        "Features",
        features,
        default=features
    )

    if not selected:
        return

    work = df[selected + [target]].dropna()

    if work[target].nunique() < 2:
        st.error("Target must contain at least two classes.")
        return

    X = work[selected]
    y = work[target].astype("category")

    class_names = list(y.cat.categories)
    y = y.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    st.metric("Test Accuracy", f"{accuracy:.2%}")

    report = classification_report(
        y_test,
        pred,
        target_names=[str(x) for x in class_names],
        output_dict=True
    )

    st.subheader("Classification Report")

    st.dataframe(
        pd.DataFrame(report).T,
        use_container_width=True
    )


def regression_page(project_key, title):
    st.header(title)

    df = upload_or_load(
        PROJECTS[project_key],
        f"{project_key}_upload"
    )

    if df is None:
        st.info("Upload your regression CSV dataset.")
        return

    dataset_info(df)

    numeric = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric) < 2:
        st.error("At least two numeric columns are required.")
        return

    target = st.selectbox(
        "Target variable",
        numeric,
        key=f"{project_key}_target"
    )

    features = [c for c in numeric if c != target]

    selected = st.multiselect(
        "Features",
        features,
        default=features,
        key=f"{project_key}_features"
    )

    if not selected:
        st.warning("Select at least one feature.")
        return

    work = df[selected + [target]].dropna()

    X = work[selected]
    y = work[target]

    if len(work) < 10:
        st.error("Not enough rows for a train/test split.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.4f}")
    c2.metric("MAE", f"{mae:.4f}")
    c3.metric("R²", f"{r2:.4f}")

    results = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": predictions
    })

    st.subheader("Predictions")
    st.dataframe(results, use_container_width=True)

    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


def deployment_page():
    st.header("🚀 Deployment Center")

    st.subheader("Install dependencies")

    st.code(
        "python -m pip install -r requirements.txt",
        language="powershell"
    )

    st.subheader("Run locally")

    st.code(
        "streamlit run app.py",
        language="powershell"
    )

    st.subheader("Project structure")

    st.code("""
streamlit_dataprofessor_series/
│
├── app.py
├── requirements.txt
├── README.md
├── 01_stock_price/
├── 02_dna_count/
├── 03_basketball_eda/
├── 04_football_eda/
├── 05_sp500_eda/
├── 06_crypto_eda/
├── 07_iris_classification/
├── 08_penguins_classification/
├── 09_boston_regression/
├── 10_solubility_regression/
├── 11_heroku_deploy/
└── 12_streamlit_sharing/
""", language="text")

    st.info(
        "For Streamlit Community Cloud, upload the complete project "
        "to GitHub and select app.py as the main application file."
    )


def sharing_page():
    st.header("🌐 Streamlit Sharing")

    st.subheader("Required files")

    st.code(
        "app.py\nrequirements.txt\nREADME.md",
        language="text"
    )

    st.subheader("Run command")

    st.code(
        "streamlit run app.py",
        language="powershell"
    )

    st.success(
        "All 12 projects can be accessed from this single dashboard."
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Data Science Dashboard")

selection = st.sidebar.radio(
    "Choose a project",
    [
        "🏠 Home",
        "📈 Stock Price",
        "🧬 DNA Count",
        "🏀 Basketball EDA",
        "⚽ Football EDA",
        "📊 S&P 500 EDA",
        "₿ Crypto EDA",
        "🌸 Iris Classification",
        "🐧 Penguins Classification",
        "🏠 Boston Regression",
        "🧪 Solubility Regression",
        "🚀 Heroku Deploy",
        "🌐 Streamlit Sharing",
    ]
)

# ============================================================
# ROUTING
# ============================================================

if selection == "🏠 Home":
    home_page()

elif selection == "📈 Stock Price":
    stock_price_page()

elif selection == "🧬 DNA Count":
    dna_page()

elif selection == "🏀 Basketball EDA":
    eda_page(
        "03 Basketball EDA",
        "🏀 Basketball Exploratory Data Analysis"
    )

elif selection == "⚽ Football EDA":
    eda_page(
        "04 Football EDA",
        "⚽ Football Exploratory Data Analysis"
    )

elif selection == "📊 S&P 500 EDA":
    eda_page(
        "05 S&P 500 EDA",
        "📊 S&P 500 Exploratory Data Analysis"
    )

elif selection == "₿ Crypto EDA":
    eda_page(
        "06 Crypto EDA",
        "₿ Cryptocurrency Exploratory Data Analysis"
    )

elif selection == "🌸 Iris Classification":
    iris_page()

elif selection == "🐧 Penguins Classification":
    penguins_page()

elif selection == "🏠 Boston Regression":
    regression_page(
        "09 Boston Regression",
        "🏠 Boston Housing Regression"
    )

elif selection == "🧪 Solubility Regression":
    regression_page(
        "10 Solubility Regression",
        "🧪 Solubility Regression"
    )

elif selection == "🚀 Heroku Deploy":
    deployment_page()

elif selection == "🌐 Streamlit Sharing":
    sharing_page()
