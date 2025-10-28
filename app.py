  # app.py
# Flood Pattern Data Mining & Forecasting - Streamlit Port of floodpatternv2.ipynb
# Interactive Plotly charts + automatic explanations below each output
# Author: ChatGPT (converted for Streamlit)
# Usage: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# ================== LOAD LOCAL CSS ==================
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
# ====================================================

st.set_page_config(layout="wide", page_title="Flood Pattern Analysis Dashboard")

# ------------------------------
# Helpers: Cleaning & Preprocess
# ------------------------------
def clean_water_level(series):
    s = series.astype(str).str.replace(' ft.', '', regex=False)\
                         .str.replace(' ft', '', regex=False)\
                         .str.replace('ft', '', regex=False)\
                         .str.replace(' ', '', regex=False)\
                         .replace('nan', pd.NA)
    s = pd.to_numeric(s, errors='coerce')
    return s

def clean_damage_col(col):
    s = col.astype(str).str.replace(',', '', regex=False)
    # fix weird patterns like '422.510.5' -> '4225105' if present
    s = s.str.replace(r'(\d)\.(\d)\.(\d)', lambda m: m.group(1)+m.group(2)+m.group(3), regex=True)
    s = pd.to_numeric(s, errors='coerce')
    return s

def _find_col(df, candidate_lower):
    """
    Return actual column name in df that matches candidate_lower (case-insensitive),
    or None if not found.
    """
    for c in df.columns:
        if c.strip().lower() == candidate_lower:
            return c
    return None

def load_and_basic_clean(df):
    # Work on a copy
    df = df.copy()

    # Normalize whitespace in column names (but keep original casing to avoid breaking other code)
    df.columns = [c.strip() for c in df.columns]

    # Create canonical column names (if any variant exists)
    # We'll create/overwrite canonical names: Year, Month, Month_Num, Day, Water Level, No. of Families affected, Damage Infrastructure, Damage Agriculture, Municipality, Barangay
    # The rest of your app expects those canonical names.
    col_map = {
        'year': _find_col(df, 'year'),
        'month': _find_col(df, 'month'),
        'month_num': _find_col(df, 'month_num'),
        'day': _find_col(df, 'day'),
        'water_level': _find_col(df, 'water level'),
        'families': _find_col(df, 'no. of families affected'),
        'damage_infra': _find_col(df, 'damage infrastructure'),
        'damage_agri': _find_col(df, 'damage agriculture'),
        'municipality': _find_col(df, 'municipality'),
        'barangay': _find_col(df, 'barangay')
    }

    # Copy found columns into canonical names (only if found)
    if col_map['year'] is not None:
        df['Year'] = df[col_map['year']]
    if col_map['month'] is not None:
        df['Month'] = df[col_map['month']].astype(str).str.strip()
    if col_map['month_num'] is not None:
        df['Month_Num'] = df[col_map['month_num']]
    if col_map['day'] is not None:
        df['Day'] = df[col_map['day']]
    if col_map['water_level'] is not None:
        df['Water Level'] = df[col_map['water_level']]
    if col_map['families'] is not None:
        df['No. of Families affected'] = df[col_map['families']]
    if col_map['damage_infra'] is not None:
        df['Damage Infrastructure'] = df[col_map['damage_infra']]
    if col_map['damage_agri'] is not None:
        df['Damage Agriculture'] = df[col_map['damage_agri']]
    if col_map['municipality'] is not None:
        df['Municipality'] = df[col_map['municipality']]
    if col_map['barangay'] is not None:
        df['Barangay'] = df[col_map['barangay']]

    # Standardize Month to uppercase names if exists
    if 'Month' in df.columns:
        df['Month'] = df['Month'].astype(str).str.strip().str.upper().replace({'NAN': pd.NA})

    # If Month_Num wasn't provided but Month names are, map names to numbers
    if 'Month_Num' not in df.columns and 'Month' in df.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        df['Month_Num'] = df['Month'].map(month_map)

    # Clean water level if present
    if 'Water Level' in df.columns:
        df['Water Level'] = clean_water_level(df['Water Level'])
        # If too many missing, leave them but otherwise impute with median
        if df['Water Level'].notna().sum() > 0:
            median_wl = df['Water Level'].median()
            df['Water Level'] = df['Water Level'].fillna(median_wl)

    # Families affected
    if 'No. of Families affected' in df.columns:
        df['No. of Families affected'] = pd.to_numeric(df['No. of Families affected'].astype(str).str.replace(',', ''), errors='coerce')
        if df['No. of Families affected'].notna().sum() > 0:
            df['No. of Families affected'] = df['No. of Families affected'].fillna(df['No. of Families affected'].median())

    # Damage columns
    for col in ['Damage Infrastructure', 'Damage Agriculture']:
        if col in df.columns:
            df[col] = clean_damage_col(df[col])
            df[col] = df[col].fillna(0)

    # Ensure Year/Month_Num/Day are numeric-ish (coerce bad ones)
    for c in ['Year','Month_Num','Day']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Try to fill forward/backward small gaps in date parts but avoid forcing wrong values:
    # Only forward/backfill when reasonable (e.g., repeated measurements across rows)
    for c in ['Year','Month_Num','Day']:
        if c in df.columns:
            # attempt forward then backward fill but only for short gaps
            df[c] = df[c].ffill().bfill()

    return df

def create_datetime_index(df):
    """
    Create a DatetimeIndex if Year/Month_Num/Day (canonical names) exist or Month name + Year + Day exist.
    Returns a dataframe with a Date index if possible; otherwise returns the original df.
    This function is robust: it coerces non-numeric parts, drops rows that still can't form valid dates,
    and avoids integer-casting errors by using pd.to_datetime with dict input.
    """
    tmp = df.copy()

    # If Month exists but Month_Num doesn't, try mapping (safe)
    if 'Month' in tmp.columns and 'Month_Num' not in tmp.columns:
        month_map = {'JANUARY':1,'FEBRUARY':2,'MARCH':3,'APRIL':4,'MAY':5,'JUNE':6,
                     'JULY':7,'AUGUST':8,'SEPTEMBER':9,'OCTOBER':10,'NOVEMBER':11,'DECEMBER':12}
        tmp['Month_Num'] = tmp['Month'].astype(str).str.strip().str.upper().map(month_map)

    # Ensure we have at least Year and something for month/day
    if not ({'Year', 'Month_Num', 'Day'}.issubset(tmp.columns)):
        # Not enough parts to build a date index
        return df

    # Coerce to numeric, leaving invalid as NaN
    tmp['Year'] = pd.to_numeric(tmp['Year'], errors='coerce')
    tmp['Month_Num'] = pd.to_numeric(tmp['Month_Num'], errors='coerce')
    tmp['Day'] = pd.to_numeric(tmp['Day'], errors='coerce')

    # Drop rows where essential parts are missing - can't build a date
    before = len(tmp)
    tmp = tmp.dropna(subset=['Year', 'Month_Num', 'Day']).copy()
    dropped = before - len(tmp)
    if dropped > 0:
        st.info(f"Dropped {dropped} rows with missing Year/Month/Day parts which couldn't form valid dates.")

    if tmp.empty:
        return df

    # Convert to integer where safe
    # (they're floats because of NaNs; cast after dropping NaNs)
    tmp['Year'] = tmp['Year'].astype(int)
    tmp['Month_Num'] = tmp['Month_Num'].astype(int)
    tmp['Day'] = tmp['Day'].astype(int)

    # Now build Date column using dict -> safe assembly
    tmp['Date'] = pd.to_datetime({'year': tmp['Year'], 'month': tmp['Month_Num'], 'day': tmp['Day']}, errors='coerce')

    # Drop rows where to_datetime still failed (e.g., Day=31 and Month=2)
    before2 = len(tmp)
    tmp = tmp.dropna(subset=['Date']).copy()
    dropped2 = before2 - len(tmp)
    if dropped2 > 0:
        st.info(f"Dropped {dropped2} rows with invalid date combinations (e.g., Feb 30).")

    if tmp.empty:
        return df

    tmp = tmp.set_index('Date').sort_index()
    return tmp

def categorize_severity(w):
    if pd.isna(w):
        return 'Unknown'
    try:
        w = float(w)
    except:
        return 'Unknown'
    if w <= 5:
        return 'Low'
    elif 5 < w <= 15:
        return 'Medium'
    else:
        return 'High'

# ------------------------------
# UI Layout
# ------------------------------
st.title("🌊 Flood Pattern Data Mining & Forecasting 🌊")
st.markdown("Upload your CSV (like `FloodDataMDRRMO.csv`) and explore the analyses. "
            "This app runs cleaning, EDA, KMeans clustering, RandomForest prediction, and SARIMA forecasting. Explanations appear under each output.")

# Sidebar: file upload & options
st.sidebar.header("Upload & Options")
uploaded_file = st.sidebar.file_uploader("Upload flood CSV", type=['csv','txt','xlsx'])
use_example = st.sidebar.checkbox("Use example dataset (if no upload)", value=False)
plotly_mode = st.sidebar.selectbox("Plot style", ["plotly (interactive)"], index=0)
show_explanations = st.sidebar.checkbox("Show explanations below outputs", value=True)

# Tabs (main)
tabs = st.tabs(["Data Upload", "Data Cleaning & EDA", "Clustering (KMeans)", "Flood Prediction (RF)", "Flood Severity", "Time Series (SARIMA)", "Tutorial"])

# ------------------------------
# 🌊 Data Upload Tab
# ------------------------------
with tabs[0]:
    st.markdown("<h2 class='main-title'>📂 Data Upload & Overview</h2>", unsafe_allow_html=True)

    # --- 1️⃣ Upload Instructions ---
    if uploaded_file is None and not use_example:
        st.info("📤 Please upload a CSV or Excel file to begin, or toggle **'Use example dataset'** in the sidebar.")
    else:
        # --- 2️⃣ Load Uploaded or Example Data ---
        if uploaded_file is not None:
            try:
                file_name = uploaded_file.name
                if file_name.endswith('.xlsx'):
                    df_raw = pd.read_excel(uploaded_file)
                else:
                    df_raw = pd.read_csv(uploaded_file)

                st.success(f"✅ Loaded **{file_name}** — **{df_raw.shape[0]:,} rows**, **{df_raw.shape[1]} columns**.")
            except Exception as e:
                st.error(f"❌ Failed to read file: {e}")
                st.stop()
        else:
            # Example dataset for demonstration
            st.warning("⚠️ Using a **synthetic example dataset** (for testing only). Upload your real file for accurate results.")
            df_raw = pd.DataFrame({
                'Year': [2018, 2018, 2019, 2019, 2020, 2020],
                'Month': ['JANUARY', 'FEBRUARY', 'DECEMBER', 'FEBRUARY', 'MAY', 'NOVEMBER'],
                'Day': [10, 5, 12, 20, 1, 15],
                'Municipality': ['Bunawan'] * 6,
                'Barangay': ['Poblacion', 'Imelda', 'Poblacion', 'Mambalili', 'Bunawan Brook', 'Poblacion'],
                'Flood Cause': ['LPA', 'LPA', 'Easterlies', 'AURING', 'Shearline', 'LPA'],
                'Water Level': ['5 ft.', '8 ft', '12ft', '20ft', 'nan', '3 ft'],
                'No. of Families affected': [10, 20, 50, 200, 0, 5],
                'Damage Infrastructure': ['0', '0', '1,000', '5,000', '0', '0'],
                'Damage Agriculture': ['0', '0', '422.510.5', '10,000', '0', '0']
            })

            # Example data preview
            st.markdown("### 🧾 Example Data Preview")
            st.dataframe(df_raw.head(), use_container_width=True)

        # --- 3️⃣ Data Summary ---
        st.markdown("### 📊 Dataset Overview")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("📅 Total Rows", f"{df_raw.shape[0]:,}")
        with info_col2:
            st.metric("📈 Total Columns", f"{df_raw.shape[1]}")

        # --- 4️⃣ Raw Data Preview (Expandable) ---
        with st.expander("🔍 View Raw Data (First 20 Rows)"):
            st.dataframe(df_raw.head(20), use_container_width=True)

        # --- 5️⃣ Column List ---
        st.markdown("### 🧩 Column Names")
        col_df = pd.DataFrame({
            "Column Name": df_raw.columns,
            "Example Value": [str(df_raw[col].iloc[0]) if not df_raw[col].empty else "" for col in df_raw.columns]
        })
        st.table(col_df)

# ------------------------------
# Cleaning & EDA Tab
# ------------------------------
with tabs[1]:
    st.header("Data Cleaning & Exploratory Data Analysis (EDA)")
    if 'df_raw' not in locals():
        st.warning("Upload a dataset first in the Data Upload tab.")
    else:
        df = load_and_basic_clean(df_raw)
        st.subheader("After basic cleaning (head):")
        st.dataframe(df.head(10))

        # Basic stats
        st.subheader("Summary statistics (numerical):")
        st.write(df.select_dtypes(include=[np.number]).describe())

        # Water Level distribution (Plotly)
        if 'Water Level' in df.columns:
            st.subheader("Water Level distribution")
            fig = px.histogram(
                df,
                x='Water Level',
                nbins=30,
                marginal="box",
                title="Distribution of Cleaned Water Level"
            )
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This histogram shows the distribution of `Water Level` after cleaning non-numeric characters
                and filling missing values with the median.  
                The boxplot margin highlights potential outliers.  
                Use this to detect skew and extreme flood events.
                """)

        # ------------------------------
        # Monthly flood probability (fixed)
        # ------------------------------
        if 'Month' in df.columns:
            # create flood_occurred column if not exists
            if 'flood_occurred' not in df.columns:
                df['flood_occurred'] = (df['Water Level'].fillna(0) > 0).astype(int)

            st.subheader("Monthly Flood Probability")

            # month mapping
            month_map = {
                1: 'January', 2: 'February', 3: 'March', 4: 'April',
                5: 'May', 6: 'June', 7: 'July', 8: 'August',
                9: 'September', 10: 'October', 11: 'November', 12: 'December'
            }

            # clean and convert month formats
            def clean_month(val):
                try:
                    val_str = str(val).strip().lower()
                    # numeric (1–12 or '01')
                    if val_str.isdigit():
                        num = int(val_str)
                        return month_map.get(num, np.nan)
                    # short text (jan, feb, mar…)
                    for num, name in month_map.items():
                        if val_str.startswith(name[:3].lower()):
                            return name
                    return np.nan
                except:
                    return np.nan

            df['Month_clean'] = df['Month'].apply(clean_month)
            df = df.dropna(subset=['Month_clean'])

            # compute monthly stats
            m_stats = df.groupby('Month_clean')['flood_occurred'].agg(['sum', 'count']).reset_index()
            m_stats['probability'] = (m_stats['sum'] / m_stats['count']).round(3)

            # keep months in correct order
            m_stats['Month_clean'] = pd.Categorical(
                m_stats['Month_clean'],
                categories=list(month_map.values()),
                ordered=True
            )
            m_stats = m_stats.sort_values('Month_clean')

            # bar chart
            fig = px.bar(
                m_stats,
                x='Month_clean',
                y='probability',
                title="Flood Probability by Month",
                text='probability'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Month", yaxis_title="Flood Probability")

            st.plotly_chart(fig, use_container_width=True)

            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This chart shows the chance of flooding per month.  
                - **Probability = Flood occurrences / Total records in that month**  
                Months with higher bars indicate higher flood risk periods.  
                """)
        # ------------------------------
        # Municipal flood probabilities
        # ------------------------------
        if 'Municipality' in df.columns:
            st.subheader("Flood probability by Municipality")
            mun = df.groupby('Municipality')['flood_occurred'].agg(['sum','count']).reset_index()
            mun['probability'] = (mun['sum'] / mun['count']).round(3)
            mun = mun.sort_values('probability', ascending=False)
            fig = px.bar(
                mun,
                x='Municipality',
                y='probability',
                title="Flood Probability by Municipality",
                text='probability'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Municipality", yaxis_title="Flood Probability")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This helps identify which municipalities historically experience more flooding,
                guiding local preparedness and response planning.
                """)
         # ------------------------------
        # Municipal flood probabilities
        # ------------------------------
        if 'Barangay' in df.columns:
            st.subheader("Flood probability by Barangay")
            mun = df.groupby('Barangay')['flood_occurred'].agg(['sum','count']).reset_index()
            mun['probability'] = (mun['sum'] / mun['count']).round(3)
            mun = mun.sort_values('probability', ascending=False)
            fig = px.bar(
                mun,
                x='Barangay',
                y='probability',
                title="Flood Probability by Barangay",
                text='probability'
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(xaxis_title="Barangay", yaxis_title="Flood Probability")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("""
                **Explanation:**  
                This helps identify which Barangay historically experience more flooding,
                guiding local preparedness and response planning.
                """)
# ------------------------------
# Clustering Tab (KMeans)
# ------------------------------
with tabs[2]:
    st.header("Clustering (KMeans)")
    if 'df' not in locals():
        st.warning("Do data cleaning first.")
    else:
        # Select features for clustering
        features = ['Water Level','No. of Families affected','Damage Infrastructure','Damage Agriculture']
        if not set(features).issubset(df.columns):
            st.error("Missing required columns for clustering.")
        else:
            st.subheader("KMeans clustering (k=3 default)")
            k = st.slider("Number of clusters (k)", 2, 6, 3)
            X_cluster = df[features].fillna(0)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster)
            df['Cluster'] = kmeans.labels_
            counts = df['Cluster'].value_counts().sort_index()
            st.write("Cluster counts:")
            st.write(counts)

            # 3d scatter (Plotly)
            fig = px.scatter_3d(df, x='Water Level', y='No. of Families affected', z='Damage Infrastructure',
                                color='Cluster', hover_data=['Barangay','Municipality','Flood Cause'],
                                title="KMeans clusters (3D)")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("**Explanation:** KMeans grouped flood events into clusters based on severity variables. Use the cluster distribution and 3D scatter to inspect which events are low vs high impact.")

            # cluster summary
            st.subheader("Cluster summary (numeric medians)")
            cluster_summary = df.groupby('Cluster')[features].median().round(2)
            st.dataframe(cluster_summary)
            if show_explanations:
                st.markdown("**Explanation:** Median values per cluster describe representative severity per cluster (water depth, families affected, damages). Useful to label clusters as 'low/medium/high' impact.")

# ------------------------------
# Flood Prediction (RandomForest) Tab
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

with tabs[3]:
    st.header("Flood occurrence prediction — RandomForest")

    if 'df' not in locals():
        st.warning("Do data cleaning first.")
    else:
        # Prepare features: water level OR numeric + month dummies
        st.markdown("We train a RandomForest to predict `flood_occurred` (binary).")

        # Create target variable
        df['flood_occurred'] = (df['Water Level'] > 0).astype(int)

        # Feature set
        month_dummies = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        X_basic = pd.concat([
            df[['Water Level', 'No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']].fillna(0),
            month_dummies
        ], axis=1)
        y = df['flood_occurred']

        # Train/test split
        Xtr, Xte, ytr, yte = train_test_split(X_basic, y, test_size=0.3, random_state=42)

        # Model training
        model = RandomForestClassifier(random_state=42)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)

        # Display header
        st.subheader("📊 Basic RandomForest Results")

        # Accuracy table
        acc_table = pd.DataFrame({
            "Metric": ["Accuracy (test)"],
            "Value": [f"{acc:.4f}"]
        })
        st.table(acc_table)

        # Classification report in tabular format
        report = classification_report(yte, ypred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)

        st.markdown("### 📈 Classification Report")
        st.table(report_df)

        # Optional explanation
        if show_explanations:
            st.markdown("""
            **🧠 Explanation:**  
            RandomForest uses many decision trees and aggregates their votes.  
            High accuracy may indicate a strong signal in the features, but always check class balance and overfitting.  
            Use the classification report to inspect precision and recall per class.
            """)

 # feature importances
        fi = pd.Series(model.feature_importances_, index=X_basic.columns).sort_values(ascending=False).head(10)
        st.subheader("Top feature importances")
        st.bar_chart(fi)

        if show_explanations:
            st.markdown("**Explanation:** Features with higher importance contributed more to model decisions. `Water Level` often dominates.")

        # Allow user to show predicted probabilities per month (as earlier notebook did)
        if st.button("Show predicted flood probability per month (using median inputs)"):
            median_vals = X_basic.median()
            months = sorted(df['Month'].dropna().unique())
            pred_rows = []
            for m in months:
                row = median_vals.copy()
                # set the month dummy for this month to 1 and others to 0 if present
                md = [c for c in X_basic.columns if c.startswith('Month_')]
                for col in md:
                    row[col] = 1 if col == f"Month_{m}" else 0
                pred_rows.append(row.values)
            Xpred = pd.DataFrame(pred_rows, columns=X_basic.columns)
            probs = model.predict_proba(Xpred)[:,1]
            prob_df = pd.DataFrame({'Month':months,'flood_prob':probs}).sort_values('flood_prob',ascending=False)
            fig = px.bar(prob_df, x='Month', y='flood_prob', title="Predicted flood probability per month (median inputs)")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("**Explanation:** This uses median numeric values and swaps month dummies to estimate flood likelihood per month. It's a model-based estimate, not a raw frequency.")

# ------------------------------
# Flood Severity Tab
# ------------------------------
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

with tabs[4]:
    st.header("🌊 Flood Severity Classification")

    if 'df' not in locals():
        st.warning("⚠️ Please perform data cleaning first.")
    else:
        # ---------------- CREATE TARGET COLUMN ----------------
        df['Flood_Severity'] = df['Water Level'].apply(categorize_severity)

        # ---------------- SEVERITY DISTRIBUTION ----------------
        st.subheader("📊 Severity Distribution")
        sev_counts = df['Flood_Severity'].value_counts().reset_index()
        sev_counts.columns = ['Severity Level', 'Count']
        st.table(sev_counts)

        # ---------------- FEATURE SETUP ----------------
        base_feats = ['No. of Families affected', 'Damage Infrastructure', 'Damage Agriculture']
        month_d = pd.get_dummies(df['Month'].astype(str).fillna('Unknown'), prefix='Month')
        muni_d = pd.get_dummies(df['Municipality'].astype(str).fillna('Unknown'), prefix='Municipality') if 'Municipality' in df.columns else pd.DataFrame()
        brgy_d = pd.get_dummies(df['Barangay'].astype(str).fillna('Unknown'), prefix='Barangay') if 'Barangay' in df.columns else pd.DataFrame()
        Xsev = pd.concat([df[base_feats].fillna(0), month_d, muni_d, brgy_d], axis=1)
        ysev = df['Flood_Severity']

        # ---------------- CLASS BALANCE TABLE ----------------
        st.subheader("⚖️ Class Counts")
        class_counts = ysev.value_counts().reset_index()
        class_counts.columns = ['Flood Severity', 'Occurrences']
        st.table(class_counts)

        # ---------------- MODEL TRAINING ----------------
        try:
            Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
                Xsev, ysev, test_size=0.3, random_state=42, stratify=ysev
            )

            model_sev = RandomForestClassifier(random_state=42)
            model_sev.fit(Xtr_s, ytr_s)
            ypred_s = model_sev.predict(Xte_s)
            acc_s = accuracy_score(yte_s, ypred_s)

            # ---------------- RESULTS TABLES ----------------
            st.subheader("✅ Severity Model Results")

            # Accuracy table
            acc_table = pd.DataFrame({
                'Metric': ['Accuracy (test)'],
                'Value': [f"{acc_s:.4f}"]
            })
            st.table(acc_table)

            # Classification report (tabular)
            report = classification_report(yte_s, ypred_s, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose().round(3)

            st.markdown("### 📈 Classification Report (Low / Medium / High)")
            st.table(report_df)

            # ---------------- EXPLANATION ----------------
            if show_explanations:
                st.markdown("""
                **🧠 Explanation:**  
                This multi-class RandomForest predicts flood severity levels — **Low**, **Medium**, or **High**.  
                Class imbalance (e.g., fewer 'High' floods) can reduce recall for rare classes.  
                For production use, consider resampling (SMOTE) or class-weight adjustments.
                """)

        except Exception as e:
            st.error(f"❌ Could not train severity model: {e}")

# ------------------------------
# Time Series (SARIMA)
# ------------------------------
with tabs[5]:
    st.header("Time Series forecasting (SARIMA)")
    if 'df' not in locals():
        st.warning("Do data cleaning first.")
    else:
        st.markdown("This section resamples Water Level to daily average, checks stationarity, fits an example SARIMA, and shows forecasts.")
        # create datetime index if possible
        df_temp = create_datetime_index(df)
        if not isinstance(df_temp.index, pd.DatetimeIndex):
            st.error("Your dataset doesn't have usable Year/Month/Day date parts to form a time index. Add Year/Month/Day columns for time series forecasting.")
        else:
            ts = df_temp['Water Level'].resample('D').mean()
            # fill NaNs for modelling
            ts_filled = ts.fillna(method='ffill').fillna(method='bfill')

            st.subheader("Time series preview (daily avg)")
            fig = px.line(ts_filled, title="Daily average Water Level")
            st.plotly_chart(fig, use_container_width=True)
            if show_explanations:
                st.markdown("**Explanation:** Original event data may have many days with no measurement; we resample to daily mean and fill gaps to produce a continuous series for SARIMA.")

            # ADF test
            st.subheader("Stationarity test (ADF)")
            try:
                adf_result = adfuller(ts_filled.dropna())
                st.write(f"ADF Statistic: {adf_result[0]:.4f}")
                st.write(f"P-value: {adf_result[1]:.4f}")
                st.write("If p-value > 0.05, series is likely non-stationary and differencing is recommended.")
            except Exception as e:
                st.error(f"ADF test failed: {e}")
                adf_result = (None, 1.0)

            if show_explanations:
                st.markdown("**Explanation:** Augmented Dickey-Fuller test checks stationarity. Non-stationary series need differencing (d>0).")

            # differencing if needed
            d = 0
            if adf_result[1] > 0.05:
                d = 1
                ts_diff = ts_filled.diff().dropna()
                fig = px.line(ts_diff, title="First-order differenced series")
                st.plotly_chart(fig, use_container_width=True)
                if show_explanations:
                    st.markdown("**Explanation:** After first differencing we remove trends; re-check ADF on differenced series before modelling.")

            # Show ACF/PACF plots
            st.subheader("ACF & PACF (help pick p/q values)")
            fig_acf = plt.figure(figsize=(10,4))
            try:
                plot_acf(ts_filled.dropna(), lags=40, ax=fig_acf.gca())
                st.pyplot(fig_acf)
            except Exception as e:
                st.error(f"ACF plot failed: {e}")
            fig_pacf = plt.figure(figsize=(10,4))
            try:
                plot_pacf(ts_filled.dropna(), lags=40, ax=fig_pacf.gca())
                st.pyplot(fig_pacf)
            except Exception as e:
                st.error(f"PACF plot failed: {e}")
            if show_explanations:
                st.markdown("**Explanation:** PACF suggests AR order (p), ACF suggests MA order (q). Seasonal spikes indicate seasonal order (P,Q,s).")

            # ------------------------------
            # Fit a sample SARIMA (table format output)
            # ------------------------------
            st.subheader("Fit example SARIMA model")
            with st.spinner("Fitting SARIMA (may take a moment)..."):
                try:
                    order = (1, d, 1)
                    seasonal_order = (1, 0, 1, 7)
                    model_sarima = SARIMAX(
                        ts_filled,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    results = model_sarima.fit(disp=False)

                    # Convert summary to table
                    summary_table = results.summary().tables[1]
                    import io
                    from pandas import read_csv
                    summary_df = read_csv(io.StringIO(summary_table.as_csv()))
                    st.dataframe(summary_df, use_container_width=True)

                    if show_explanations:
                        st.markdown("""
                        **Explanation:**  
                        SARIMA models capture both seasonal and non-seasonal components.  
                        This table summarizes coefficient estimates, standard errors, z-values, and p-values.  
                        Use AIC/BIC for model comparison.
                        """)
                except Exception as e:
                    st.error(f"SARIMA fit failed: {e}")
                    results = None

            # Forecast
            steps = st.slider("Forecast horizon (days)", 7, 365, 30)
            try:
                if results is not None:
                    pred = results.get_forecast(steps=steps)
                    pred_mean = pred.predicted_mean
                    pred_ci = pred.conf_int()

                    # Plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts_filled.index, y=ts_filled, name='Observed'))
                    fig.add_trace(go.Scatter(x=pred_mean.index, y=pred_mean, name='Forecast'))
                    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,0], fill=None, mode='lines', line=dict(width=0)))
                    fig.add_trace(go.Scatter(x=pred_ci.index, y=pred_ci.iloc[:,1], fill='tonexty', name='95% CI', mode='lines', line=dict(width=0)))
                    fig.update_layout(title="SARIMA Forecast", xaxis_title="Date", yaxis_title="Water Level")
                    st.plotly_chart(fig, use_container_width=True)

                    if show_explanations:
                        st.markdown("""
                        **Explanation:**  
                        The forecast shows predicted mean water levels with 95% confidence intervals.  
                        Ideal for short-term flood preparedness planning.
                        """)
                else:
                    st.error("No SARIMA results available to forecast.")
            except Exception as e:
                st.error(f"Forecast failed: {e}")


# ------------------------------
# Tutorial Tab
# ------------------------------
with tabs[6]:
    st.header("Tutorial & Walkthrough")
    st.markdown("""
    This tutorial explains the pipeline and what each section does.

    ### 1. Data Upload
    - Upload your CSV file (e.g., `FloodDataMDRRMO.csv`) containing columns like:
      `Year, Month, Day, Municipality, Barangay, Flood Cause, Water Level, No. of Families affected, Damage Infrastructure, Damage Agriculture`.
    - If any column names differ, adapt the column name references in the script.

    ### 2. Data Cleaning
    - `Water Level` cleaned from text like "5 ft." → numeric.
    - `Damage` columns cleaned by removing commas and converting to numeric.
    - Missing numeric values are imputed (median or 0 depending on the column).
    - `flood_occurred` is derived as `Water Level > 0`.

    **Tip:** Real datasets may require extra cleaning for typos/fuzzy entries.

    ### 3. Exploratory Data Analysis (EDA)
    - Water Level distribution (Histogram + boxplot).
    - Monthly and municipal flood probabilities calculated as (#flooding rows)/(#rows per group).
    - These help identify peak months and hotspots.

    ### 4. KMeans Clustering
    - Clusters the flood events using `Water Level`, `No. of Families affected`, and damage columns.
    - Use clusters to label events (e.g., low/medium/high impact).
    - Check cluster medians to interpret cluster meaning.

    ### 5. RandomForest Flood Occurrence Prediction
    - Trains a RandomForest to predict whether a flood occurs (binary) using numeric + month dummies.
    - Outputs accuracy and classification report.
    - Also shows feature importances.

    **Caveats:** If accuracy is suspiciously high, check for leakage in the data or extremely imbalanced classes.

    ### 6. Flood Severity Classification
    - Categorizes severity from Water Level (Low/Medium/High).
    - Trains a multi-class RandomForest. Imbalanced classes may need resampling or class weights.

    ### 7. Time Series (SARIMA)
    - Requires date components `Year`, `Month`, `Day` to create a datetime index.
    - Resamples daily, fills missing values, checks stationarity (ADF), inspects ACF/PACF, fits example SARIMA, and produces forecasts.
    - Tune SARIMA orders via grid search and diagnostic checks for production.

    ### 8. Reproducibility & Deployment
    - Use the included `requirements.txt` to create a virtualenv.
    - Run: `streamlit run app.py`.
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("App converted from Colab -> Streamlit. If you want, I can:")
st.sidebar.markdown("- Add model persistence (save/load trained models)\n- Add resampling for imbalance (SMOTE/oversample)\n- Add downloadable reports (PDF/Excel)\n\nIf you want any of those, say the word and I'll add it.")
