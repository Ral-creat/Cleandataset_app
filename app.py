import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import re
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Streamlit Imports ---
try:
    import streamlit as st
    st.set_page_config(layout="wide", page_title="Bunawan Flood Pattern Analysis")
    is_streamlit = True
except ImportError:
    is_streamlit = False

# --- Configuration ---
warnings.filterwarnings("ignore")
CSV_PATH = "cleaned_flood_data.csv"
OUTDIR = "." # Use current directory for file saving

# --- Helper Functions ---
def find_col(df_cols, keywords):
    """Detects likely columns by keywords."""
    df_cols_lower = [c.lower() for c in df_cols]
    for k in keywords:
        for i, c in enumerate(df_cols_lower):
            if k in c:
                return df_cols[i]
    return None

def clean_water_level(level):
    """Extracts the first numeric value from the water level string."""
    if pd.isna(level) or str(level).strip() in ['0', '0.0', '0ft.', '0 ft.', '0/ 0', '0/0', '0.0/0.0']:
        return 0.0
    match = re.search(r'(\d+\.?\d*)', str(level))
    if match:
        return float(match.group(1))
    return 0.0

def clean_damage_value(value):
    """Robustly cleans string values (e.g., '10,000,000.00') to numeric."""
    if pd.isna(value):
        return 0.0
    s = str(value).replace(',', '') # Remove commas
    return pd.to_numeric(re.sub(r'[^\d.]', '', s), errors='coerce')


# --- Main Data Processing and Plotting ---
def run_analysis(csv_path):
    if not os.path.exists(csv_path):
        if is_streamlit:
            st.error(f"File not found: {csv_path}. Please make sure 'cleaned_flood_data.csv' is in the same directory.")
        else:
            print(f"Error: File not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False) # Fallback to default encoding

    # --- Column Detection ---
    df_cols = df.columns.tolist()
    date_col = find_col(df_cols, ['date','datetime','time'])
    day_col = find_col(df_cols, ['day'])
    year_col = find_col(df_cols, ['year'])
    water_col = find_col(df_cols, ['water','level','wl','depth','height'])
    area_col = find_col(df_cols, ['barangay','brgy','area','location','sitio'])
    flood_cause_col = find_col(df_cols, ['flood cause', 'cause'])
    damage_inf_col = find_col(df_cols, ['infrastruct','infra','building'])
    damage_agri_col = find_col(df_cols, ['agri','agriculture','crop','farm'])
    damage_any_col = find_col(df_cols, ['damage','loss','estimated_damage','total_damage'])

    # --- Date Parsing and Indexing ---
    if date_col and day_col and year_col:
        df['__combined_date'] = df[date_col].astype(str) + ' ' + df[day_col].astype(str) + ', ' + df[year_col].astype(str)
        temp_date_col = '__combined_date'
    else:
        # Fallback for inconsistent date format/columns
        temp_date_col = date_col if date_col else 'Date' # Assuming 'Date' still holds month info

    df[temp_date_col] = pd.to_datetime(df[temp_date_col], errors='coerce', infer_datetime_format=True)
    df = df.sort_values(by=temp_date_col).reset_index(drop=True)
    df = df.set_index(pd.DatetimeIndex(df[temp_date_col]))
    df = df.dropna(subset=[temp_date_col])

    # --- Water Level Handling ---
    if water_col is None:
        raise ValueError("No water-level column found. Please check column names.")

    df['Cleaned_Water_Level'] = df[water_col].apply(clean_water_level)
    water_col = 'Cleaned_Water_Level'
    df[water_col] = df[water_col].interpolate(method='linear', limit_direction='both')

    # --- Flood Heuristic ---
    df['zscore_water'] = stats.zscore(df[water_col].fillna(df[water_col].mean()))
    threshold = df[water_col].mean() + 1.0 * df[water_col].std()
    df['is_flood'] = (df[water_col] >= threshold) | (df['zscore_water'].abs() > 1.5)
    df['year'] = df.index.year

    # --- Damage Columns ---
    damage_cols_raw = [damage_inf_col, damage_agri_col, damage_any_col]
    damage_cols = [c for c in damage_cols_raw if c is not None and c in df.columns]
    damage_cols = list(dict.fromkeys(damage_cols))

    if damage_cols:
        for c in damage_cols:
            df[c] = df[c].apply(clean_damage_value)
        total_damage_per_year = df.groupby('year')[damage_cols].sum().fillna(0)
    else:
        total_damage_per_year = pd.DataFrame()

    # --- Aggregations ---
    floods_per_year = df.groupby('year')['is_flood'].sum().astype(int)
    avg_water_per_year = df.groupby('year')[water_col].mean()

    if area_col and area_col in df.columns:
        most_affected_by_year = df[df['is_flood']].groupby(['year', area_col]).size().reset_index(name='Flood_Count')
        # Find the most affected barangay each year
        idx = most_affected_by_year.groupby(['year'])['Flood_Count'].transform(max) == most_affected_by_year['Flood_Count']
        most_affected_barangays = most_affected_by_year[idx].sort_values(by='year')
    else:
        most_affected_barangays = pd.DataFrame()

    # --- PLOTS ---

    # 1. Monthly Flood Occurrence Heatmap (Request 1)
    df['month'] = df.index.month
    monthly_yearly_counts = df.groupby(['year', 'month']).size().reset_index(name='Flood_Count')
    monthly_yearly_counts['Month_Name'] = monthly_yearly_counts['month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%b'))

    pivot_table = monthly_yearly_counts.pivot(index='Month_Name', columns='year', values='Flood_Count').fillna(0)
    month_order_names = [pd.to_datetime(str(i), format='%m').strftime('%b') for i in range(1, 13)]
    pivot_table = pivot_table.reindex(month_order_names, axis=0)

    plt.figure(figsize=(12, 6))
    plt.imshow(pivot_table, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Number of Flood Events')
    plt.yticks(ticks=np.arange(len(pivot_table.index)), labels=pivot_table.index)
    plt.xticks(ticks=np.arange(len(pivot_table.columns)), labels=pivot_table.columns)
    plt.title('Monthly Flood Occurrence Heatmap by Year (2014-2025)')
    plt.xlabel('Year')
    plt.ylabel('Month')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "monthly_flood_occurrence_heatmap.png"))
    plt.close() # Close plot for Streamlit

    # 2. Most Affected Barangays (Request 2)
    plt.figure(figsize=(12, 5))
    if not most_affected_barangays.empty:
        # Show top 10 most affected across ALL years
        overall_most_affected = df[df['is_flood']].groupby(area_col).size().sort_values(ascending=False).head(10)
        overall_most_affected.plot(kind='bar')
        plt.title(f"Top 10 Overall Affected Areas ({area_col}) by Flood Count")
        plt.xlabel(area_col); plt.ylabel("Total Flood Count")
        plt.xticks(rotation=45, ha='right')
    else:
        plt.text(0.5, 0.5, "No Area Data or Flood Events Found", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "most_affected_areas_overall.png"))
    plt.close()

    # 3. Average Water Level per Year (Request 3)
    plt.figure(figsize=(8, 4))
    avg_water_per_year.plot(kind='bar')
    plt.title("Average Water Level per Year")
    plt.xlabel("Year"); plt.ylabel(f"Average {water_col} (First Value Extracted)")
    plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"avg_water_per_year.png"))
    plt.close()

    # 4. Common Cause of Flood (Request 4)
    if flood_cause_col and flood_cause_col in df.columns:
        flood_causes_counts = df[flood_cause_col].value_counts().head(10)
        plt.figure(figsize=(10, 5))
        flood_causes_counts.sort_values().plot(kind='barh')
        plt.title("Top 10 Most Common Flood Causes")
        plt.xlabel("Count"); plt.ylabel(flood_cause_col)
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "top_flood_causes.png"))
        plt.close()

    # 5. Total Damage per Year
    if not total_damage_per_year.empty:
        plt.figure(figsize=(8,4))
        for c in total_damage_per_year.columns:
            plt.plot(total_damage_per_year.index, total_damage_per_year[c], marker='o', label=c)
        plt.title("Total Damage per Year")
        plt.xlabel("Year"); plt.ylabel("Damage (PhP)") # Assuming Philippine Pesos as common currency
        plt.legend(title='Damage Type')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"total_damage_per_year.png"))
        plt.close()

    # 6. SARIMA Time Series
    series = df[water_col].resample('M').mean().fillna(0)
    if len(series) >= 12:
        # Simplified SARIMA for Streamlit (to avoid long processing)
        if is_streamlit:
            p, d, q = 1, 0, 0; P, D, Q = 1, 0, 0
            order = (p,d,q); s_order = (P,D,Q,12)
        else:
            # Use the best order found in the VM step: ((0, 0, 0), (0, 0, 1, 12))
            p, d, q = 0, 0, 0; P, D, Q = 0, 0, 1
            order = (p,d,q); s_order = (P,D,Q,12)

        split = int(len(series)*0.8)
        train = series.iloc[:split]; test = series.iloc[split:]
        try:
            mod = SARIMAX(train, order=order, seasonal_order=s_order,
                          enforce_stationarity=False, enforce_invertibility=False)
            res = mod.fit(disp=False)
            pred = res.get_prediction(start=test.index[0], end=test.index[-1], dynamic=False)
            forecast = pred.predicted_mean
            mae = mean_absolute_error(test, forecast); mse = mean_squared_error(test, forecast)

            plt.figure(figsize=(10,4))
            plt.plot(train.index, train, label='Train')
            plt.plot(test.index, test, label='Test')
            plt.plot(forecast.index, forecast, label=f'SARIMA Forecast {order}x{s_order}')
            plt.legend(); plt.title("SARIMA: Actual vs Forecast (Water Level)")
            plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,"sarima_actual_vs_forecast.png"))
            plt.close()
        except Exception as e:
            print(f"SARIMA failed in app: {e}")

    # --- Streamlit Display ---
    if is_streamlit:
        st.title("🌊 Bunawan Flood Pattern Analysis (2014-2025)")
        st.markdown("---")

        st.header("1. Flood Pattern and Occurrence")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Flood Occurrence Heatmap")
            st.image(os.path.join(OUTDIR, "monthly_flood_occurrence_heatmap.png"))
        with col2:
            st.subheader("Flood Occurrences per Year")
            st.image(os.path.join(OUTDIR, "floods_per_year.png"))
            st.dataframe(floods_per_year.rename("Flood Count").to_frame())


        st.markdown("---")
        st.header("2. Water Level and Flood Event Markers")
        st.image(os.path.join(OUTDIR, "water_level_with_flood_markers.png"))

        st.markdown("---")
        st.header("3. Affected Areas and Flood Causes")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Top Affected Barangays")
            st.image(os.path.join(OUTDIR, "most_affected_areas_overall.png"))
            if not most_affected_barangays.empty:
                st.markdown("**Most Affected Barangay per Year:**")
                st.dataframe(most_affected_barangays.set_index('year'))

        with col4:
            st.subheader("Top 10 Common Flood Causes")
            if os.path.exists(os.path.join(OUTDIR, "top_flood_causes.png")):
                st.image(os.path.join(OUTDIR, "top_flood_causes.png"))
            else:
                st.
