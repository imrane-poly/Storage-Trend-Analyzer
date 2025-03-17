# Required Libraries for Storage Trend Analyzer

# Basic libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Data visualization
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistical analysis and forecasting
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="OilX Storage Trend Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning {
        color: #ff4b4b;
        font-weight: bold;
    }
    .success {
        color: #00c853;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def load_data():
    try:
        terminals = pd.read_csv("Terminals_Sampled_24MB.csv")
        terminals_weekly = pd.read_csv("TerminalsWeekly.2025-02-19T09-55.csv")
        floating_storage = pd.read_csv("FloatingStorageVessels.2025-02-19T03-30.csv")
        
        # Convert dates for all dataframes
        for df in [terminals, terminals_weekly, floating_storage]:
            date_columns = [col for col in df.columns if 'Date' in col]
            for date_col in date_columns:
                df[date_col] = pd.to_datetime(df[date_col])
        
        # For demo, use sample data if needed
        if len(terminals) == 0:
            terminals = pd.read_csv("Representative_Sample_of_Terminals_Data.csv")
            terminals_weekly = pd.read_csv("Representative_Sample_of_Terminals_Weekly_Data.csv")
            floating_storage = pd.read_csv("Representative_Sample_of_Floating_Storage_Vessels_Data.csv")
            
            # Convert dates again for sample data
            for df in [terminals, terminals_weekly, floating_storage]:
                date_columns = [col for col in df.columns if 'Date' in col]
                for date_col in date_columns:
                    df[date_col] = pd.to_datetime(df[date_col])
                    
        # Additional data cleaning
        # Filter for relevant flow breakdown for consistent analysis
        terminals = terminals[terminals['FlowBreakdown'].str.contains('Total Stocks|Stocks', case=False, na=False)]
        terminals_weekly = terminals_weekly[terminals_weekly['FlowBreakdown'].str.contains('Total Stocks|Stocks', case=False, na=False)]
        
        # Filter out zero or negative values that might represent missing data
        terminals = terminals[terminals['ObservedValue'] > 0]
        terminals_weekly = terminals_weekly[terminals_weekly['ObservedValue'] > 0]
        floating_storage = floating_storage[floating_storage['QuantityKiloBarrels'] > 0]
        
        return terminals, terminals_weekly, floating_storage
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def preprocess_data(terminals, terminals_weekly, floating_storage):
    # Create more sophisticated aggregation with error handling
    try:
        # Terminals aggregation by country and date
        terminals_agg = terminals.groupby(["CountryName", "ReferenceDate"])["ObservedValue"].sum().reset_index()
        
        # Terminal aggregation by terminal name for detailed view
        terminal_level_agg = terminals.groupby(["TerminalName", "CountryName", "ReferenceDate"])["ObservedValue"].sum().reset_index()
        
        # Floating storage aggregation
        floating_storage_agg = floating_storage.groupby(["OriginCountry", "ReferenceDate"])["QuantityKiloBarrels"].sum().reset_index()
        floating_storage_agg.rename(columns={"QuantityKiloBarrels": "FloatingStorage_KiloBarrels", "OriginCountry": "CountryName"}, inplace=True)
        
        # Create area-based floating storage aggregation for geographical insights
        area_storage_agg = floating_storage.groupby(["Area", "ReferenceDate"])["QuantityKiloBarrels"].sum().reset_index()
        
        # Vessel class aggregation for fleet analysis
        vessel_class_agg = floating_storage.groupby(["VesselClass", "ReferenceDate"])["QuantityKiloBarrels"].sum().reset_index()
        
        # Merge data with proper handling for missing values
        # First create complete country-date combinations to ensure time series continuity
        all_countries = terminals_agg["CountryName"].unique()
        min_date = min(terminals_agg["ReferenceDate"].min(), floating_storage_agg["ReferenceDate"].min())
        max_date = max(terminals_agg["ReferenceDate"].max(), floating_storage_agg["ReferenceDate"].max())
        
        # Create date range with monthly frequency
        date_range = pd.date_range(start=min_date, end=max_date, freq='MS')
        
        # Create complete grid of countries and dates
        country_date_grid = pd.DataFrame([(country, date) for country in all_countries for date in date_range],
                                        columns=["CountryName", "ReferenceDate"])
        
        # Merge with terminals data
        merged_data = pd.merge(country_date_grid, terminals_agg, 
                              on=["CountryName", "ReferenceDate"], how="left")
        
        # Merge with floating storage data
        storage_data = pd.merge(merged_data, floating_storage_agg,
                               on=["CountryName", "ReferenceDate"], how="left")
        
        # Fill missing values
        storage_data["ObservedValue"] = storage_data["ObservedValue"].fillna(0)
        storage_data["FloatingStorage_KiloBarrels"] = storage_data["FloatingStorage_KiloBarrels"].fillna(0)
        
        # Calculate total storage
        storage_data["TotalStorage_KiloBarrels"] = storage_data["ObservedValue"] + storage_data["FloatingStorage_KiloBarrels"]
        
        # Calculate storage changes for trend analysis
        storage_data["StorageChange"] = storage_data.groupby("CountryName")["TotalStorage_KiloBarrels"].diff()
        
        # Calculate storage utilization (assuming max historical is proxy for capacity)
        country_max_storage = storage_data.groupby("CountryName")["TotalStorage_KiloBarrels"].max()
        storage_data["StorageCapacity"] = storage_data["CountryName"].map(country_max_storage)
        storage_data["UtilizationRate"] = (storage_data["TotalStorage_KiloBarrels"] / storage_data["StorageCapacity"]) * 100
        
        # Calculate 3-month and 6-month average for trend comparison
        storage_data["3M_Avg"] = storage_data.groupby("CountryName")["TotalStorage_KiloBarrels"].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        storage_data["6M_Avg"] = storage_data.groupby("CountryName")["TotalStorage_KiloBarrels"].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate Year-over-Year change
        storage_data["YoY_Change"] = storage_data.groupby("CountryName")["TotalStorage_KiloBarrels"].pct_change(periods=12) * 100
        
        return storage_data, terminal_level_agg, area_storage_agg, vessel_class_agg
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load datasets
with st.spinner("Loading data..."):
    terminals, terminals_weekly, floating_storage = load_data()
    storage_data, terminal_level_agg, area_storage_agg, vessel_class_agg = preprocess_data(terminals, terminals_weekly, floating_storage)

# Check if data is successfully loaded
if storage_data.empty:
    st.error("Failed to load or process data. Please check your data files.")
    st.stop()

# Sidebar for navigation and filters
st.sidebar.markdown('<div class="main-header">Storage Trend Analyzer</div>', unsafe_allow_html=True)

# Add logo if available
# st.sidebar.image("gunvor_logo.png", width=200)

# Navigation
page = st.sidebar.radio("Navigation", 
    ["Dashboard Overview", "Regional Analysis", "Terminal Details", 
     "Floating Storage", "Forecasting", "Anomaly Detection"])

# Date range filter
min_date = storage_data["ReferenceDate"].min().date()
max_date = storage_data["ReferenceDate"].max().date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(max_date - timedelta(days=365), max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_data = storage_data[(storage_data["ReferenceDate"].dt.date >= start_date) & 
                                (storage_data["ReferenceDate"].dt.date <= end_date)]
else:
    filtered_data = storage_data

# Region filter
all_regions = sorted(storage_data["CountryName"].unique())
selected_regions = st.sidebar.multiselect("Select Regions", all_regions, default=all_regions[:5])

if not selected_regions:
    st.sidebar.warning("Please select at least one region.")
    selected_regions = all_regions[:1]

filtered_data = filtered_data[filtered_data["CountryName"].isin(selected_regions)]

# Dashboard Overview Page
if page == "Dashboard Overview":
    st.markdown('<div class="main-header">Storage Trend Dashboard</div>', unsafe_allow_html=True)
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest date with data
    latest_date = filtered_data["ReferenceDate"].max()
    latest_data = filtered_data[filtered_data["ReferenceDate"] == latest_date]
    
    with col1:
        total_current_storage = latest_data["TotalStorage_KiloBarrels"].sum()
        st.metric("Total Storage (KB)", f"{total_current_storage:,.0f}")
    
    with col2:
        terminal_storage = latest_data["ObservedValue"].sum()
        st.metric("Terminal Storage (KB)", f"{terminal_storage:,.0f}")
    
    with col3:
        floating_storage_val = latest_data["FloatingStorage_KiloBarrels"].sum()
        st.metric("Floating Storage (KB)", f"{floating_storage_val:,.0f}")
    
    with col4:
        # Calculate MoM change
        prev_month = latest_date - pd.DateOffset(months=1)
        prev_month_data = filtered_data[filtered_data["ReferenceDate"] == prev_month]
        
        if not prev_month_data.empty:
            prev_storage = prev_month_data["TotalStorage_KiloBarrels"].sum()
            mom_change = ((total_current_storage - prev_storage) / prev_storage * 100) if prev_storage > 0 else 0
            st.metric("MoM Change", f"{mom_change:.1f}%", delta=f"{mom_change:.1f}%")
        else:
            st.metric("MoM Change", "N/A")
    
    # Main storage trend chart
    st.markdown('<div class="sub-header">Global Storage Trends</div>', unsafe_allow_html=True)
    
    # Aggregate data by date for the global trend
    global_trend = filtered_data.groupby("ReferenceDate")[["ObservedValue", "FloatingStorage_KiloBarrels", "TotalStorage_KiloBarrels"]].sum().reset_index()
    
    # Create interactive plotly chart
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add terminal storage as bar
    fig.add_trace(
        go.Bar(
            x=global_trend["ReferenceDate"], 
            y=global_trend["ObservedValue"],
            name="Terminal Storage",
            marker_color="#1E3A8A"
        )
    )
    
    # Add floating storage as bar
    fig.add_trace(
        go.Bar(
            x=global_trend["ReferenceDate"], 
            y=global_trend["FloatingStorage_KiloBarrels"],
            name="Floating Storage",
            marker_color="#4F46E5"
        )
    )
    
    # Add total storage line
    fig.add_trace(
        go.Scatter(
            x=global_trend["ReferenceDate"], 
            y=global_trend["TotalStorage_KiloBarrels"],
            name="Total Storage",
            line=dict(color="#EF4444", width=2),
            mode="lines"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Global Storage Trends",
        barmode="stack",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig.update_yaxes(
        title_text="Storage (KiloBarrels)",
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="Total Storage (KiloBarrels)",
        secondary_y=True,
        showgrid=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top 10 Countries by Current Storage
    st.markdown('<div class="sub-header">Top Countries by Current Storage</div>', unsafe_allow_html=True)
    
    top_countries = latest_data.sort_values("TotalStorage_KiloBarrels", ascending=False).head(10)
    
    fig = px.bar(
        top_countries,
        x="CountryName",
        y="TotalStorage_KiloBarrels",
        color="TotalStorage_KiloBarrels",
        color_continuous_scale="Viridis",
        text=top_countries["TotalStorage_KiloBarrels"].round(0).astype(int)
    )
    
    fig.update_layout(
        title="Top 10 Countries by Current Storage",
        xaxis_title="Country",
        yaxis_title="Total Storage (KiloBarrels)",
        height=400
    )
    
    fig.update_traces(texttemplate='%{text:,}', textposition='outside')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Countries with Largest Changes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="sub-header">Largest Storage Builds (Last Month)</div>', unsafe_allow_html=True)
        
        # Calculate monthly change
        monthly_change = filtered_data.copy()
        monthly_change["PrevMonth"] = monthly_change.groupby("CountryName")["TotalStorage_KiloBarrels"].shift(1)
        monthly_change["MonthlyChange"] = monthly_change["TotalStorage_KiloBarrels"] - monthly_change["PrevMonth"]
        
        # Get latest month data
        latest_month_data = monthly_change[monthly_change["ReferenceDate"] == latest_date]
        
        # Top 5 builds
        top_builds = latest_month_data.sort_values("MonthlyChange", ascending=False).head(5)
        
        if not top_builds.empty:
            fig = px.bar(
                top_builds,
                x="CountryName",
                y="MonthlyChange",
                color="MonthlyChange",
                color_continuous_scale="Greens",
                text=top_builds["MonthlyChange"].round(0).astype(int)
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Monthly Change (KiloBarrels)",
                height=300
            )
            
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data to calculate monthly changes.")
    
    with col2:
        st.markdown('<div class="sub-header">Largest Storage Draws (Last Month)</div>', unsafe_allow_html=True)
        
        # Top 5 draws
        top_draws = latest_month_data.sort_values("MonthlyChange").head(5)
        
        if not top_draws.empty:
            fig = px.bar(
                top_draws,
                x="CountryName",
                y="MonthlyChange",
                color="MonthlyChange",
                color_continuous_scale="Reds_r",
                text=top_draws["MonthlyChange"].round(0).astype(int)
            )
            
            fig.update_layout(
                xaxis_title="Country",
                yaxis_title="Monthly Change (KiloBarrels)",
                height=300
            )
            
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data to calculate monthly changes.")
    
    # Regions Approaching Capacity Constraints
    st.markdown('<div class="sub-header">Regions Approaching Capacity</div>', unsafe_allow_html=True)
    
    # Define threshold for capacity constraint warning (e.g., 85% of max historical)
    capacity_threshold = 0.85
    
    # Calculate utilization
    latest_data["UtilizationRate"] = latest_data["TotalStorage_KiloBarrels"] / latest_data["StorageCapacity"] * 100
    
    # Filter for regions approaching capacity
    approaching_capacity = latest_data[latest_data["UtilizationRate"] >= capacity_threshold * 100].sort_values("UtilizationRate", ascending=False)
    
    if not approaching_capacity.empty:
        fig = px.bar(
            approaching_capacity,
            x="CountryName",
            y="UtilizationRate",
            color="UtilizationRate",
            color_continuous_scale="RdYlGn_r",
            text=approaching_capacity["UtilizationRate"].round(1)
        )
        
        fig.update_layout(
            title="Regions Approaching Storage Capacity",
            xaxis_title="Country",
            yaxis_title="Utilization Rate (%)",
            height=400
        )
        
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=90,
            x1=len(approaching_capacity) - 0.5,
            y1=90,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table with more details
        st.dataframe(
            approaching_capacity[["CountryName", "TotalStorage_KiloBarrels", "StorageCapacity", "UtilizationRate"]]
            .rename(columns={
                "TotalStorage_KiloBarrels": "Current Storage (KB)",
                "StorageCapacity": "Estimated Capacity (KB)",
                "UtilizationRate": "Utilization (%)"
            })
            .set_index("CountryName")
            .style.format({
                "Current Storage (KB)": "{:,.0f}",
                "Estimated Capacity (KB)": "{:,.0f}",
                "Utilization (%)": "{:.1f}%"
            })
            .background_gradient(cmap="YlOrRd", subset=["Utilization (%)"])
        )
    else:
        st.info("No regions currently approaching capacity.")

# Regional Analysis Page
elif page == "Regional Analysis":
    st.markdown('<div class="main-header">Regional Storage Analysis</div>', unsafe_allow_html=True)
    
    # Region selector
    selected_region = st.selectbox("Select a Region for Detailed Analysis", selected_regions)
    
    # Filter data for selected region
    region_data = filtered_data[filtered_data["CountryName"] == selected_region]
    
    if region_data.empty:
        st.warning(f"No data available for {selected_region}.")
        st.stop()
    
    # Regional overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest date with data for this region
    latest_date = region_data["ReferenceDate"].max()
    latest_region_data = region_data[region_data["ReferenceDate"] == latest_date]
    
    with col1:
        current_storage = latest_region_data["TotalStorage_KiloBarrels"].sum()
        st.metric("Current Storage (KB)", f"{current_storage:,.0f}")
    
    with col2:
        # Calculate YoY change
        yoy_change = latest_region_data["YoY_Change"].values[0] if not latest_region_data.empty else None
        
        if yoy_change is not None and not np.isnan(yoy_change):
            st.metric("YoY Change", f"{yoy_change:.1f}%", delta=f"{yoy_change:.1f}%")
        else:
            st.metric("YoY Change", "N/A")
    
    with col3:
        # Get utilization rate
        utilization = latest_region_data["UtilizationRate"].values[0] if not latest_region_data.empty else None
        
        if utilization is not None and not np.isnan(utilization):
            st.metric("Capacity Utilization", f"{utilization:.1f}%")
        else:
            st.metric("Capacity Utilization", "N/A")
    
    with col4:
        # Calculate average change over last 3 months
        last_3m = region_data.sort_values("ReferenceDate", ascending=False).head(3)
        
        if len(last_3m) >= 2:
            avg_change = last_3m["StorageChange"].mean()
            st.metric("Avg Monthly Change (KB)", f"{avg_change:,.0f}", 
                     delta=f"{avg_change:,.0f}", delta_color="normal")
        else:
            st.metric("Avg Monthly Change (KB)", "N/A")
    
    # Historical trend chart with components
    st.markdown('<div class="sub-header">Storage Trend Analysis</div>', unsafe_allow_html=True)
    
    # Create interactive plotly chart with terminal and floating storage components
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add terminal storage as bar
    fig.add_trace(
        go.Bar(
            x=region_data["ReferenceDate"], 
            y=region_data["ObservedValue"],
            name="Terminal Storage",
            marker_color="#1E3A8A"
        )
    )
    
    # Add floating storage as bar
    fig.add_trace(
        go.Bar(
            x=region_data["ReferenceDate"], 
            y=region_data["FloatingStorage_KiloBarrels"],
            name="Floating Storage",
            marker_color="#4F46E5"
        )
    )
    
    # Add total storage line
    fig.add_trace(
        go.Scatter(
            x=region_data["ReferenceDate"], 
            y=region_data["TotalStorage_KiloBarrels"],
            name="Total Storage",
            line=dict(color="#EF4444", width=2),
            mode="lines"
        ),
        secondary_y=True
    )
    
    # Add 6-month moving average
    fig.add_trace(
        go.Scatter(
            x=region_data["ReferenceDate"], 
            y=region_data["6M_Avg"],
            name="6-Month Average",
            line=dict(color="#059669", width=2, dash="dash"),
            mode="lines"
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"Storage Trends for {selected_region}",
        barmode="stack",
        xaxis_title="Date",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    fig.update_yaxes(
        title_text="Component Storage (KiloBarrels)",
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="Total Storage (KiloBarrels)",
        secondary_y=True,
        showgrid=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly change analysis
    st.markdown('<div class="sub-header">Monthly Storage Changes</div>', unsafe_allow_html=True)
    
    # Filter out missing values for the chart
    changes_data = region_data.dropna(subset=["StorageChange"])
    
    fig = px.bar(
        changes_data,
        x="ReferenceDate",
        y="StorageChange",
        color=np.where(changes_data["StorageChange"] >= 0, "Increase", "Decrease"),
        color_discrete_map={"Increase": "#059669", "Decrease": "#DC2626"},
        title=f"Monthly Storage Changes for {selected_region}"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Change (KiloBarrels)",
        showlegend=True,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.markdown('<div class="sub-header">Seasonal Storage Patterns</div>', unsafe_allow_html=True)
    
    # Extract month and calculate monthly averages
    region_data["Month"] = region_data["ReferenceDate"].dt.month
    
    # Calculate average storage by month
    seasonal_data = region_data.groupby("Month")["TotalStorage_KiloBarrels"].mean().reset_index()
    
    # Map month numbers to names
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    seasonal_data["Month_Name"] = seasonal_data["Month"].map(month_names)
    
    fig = px.line(
        seasonal_data,
        x="Month_Name",
        y="TotalStorage_KiloBarrels",
        markers=True,
        title=f"Average Storage by Month for {selected_region}"
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Average Storage (KiloBarrels)",
        height=400
    )
    
    # Ensure months are displayed in correct order
    fig.update_xaxes(categoryorder='array', 
                    categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal decomposition if enough data
    if len(region_data) >= 24:  # Need at least 2 years of data
        st.markdown('<div class="sub-header">Seasonal Decomposition Analysis</div>', unsafe_allow_html=True)
        
        try:
            # Set index to date for time series processing
            ts_data = region_data.set_index("ReferenceDate")["TotalStorage_KiloBarrels"]
            
            # Fill gaps if any
            ts_data = ts_data.asfreq('MS', method='ffill')  # Monthly start frequency
            
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            
            # Create figure with subplots
            fig = make_subplots(rows=4, cols=1, 
                              subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
            
            # Add traces
            fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values, mode='lines', name='Observed'), row=1, col=1)
            fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, mode='lines', name='Trend'), row=2, col=1)
            fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, mode='lines', name='Seasonal'), row=3, col=1)
            fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, mode='lines', name='Residual'), row=4, col=1)
            
            # Update layout
            fig.update_layout(height=800, title_text=f"Seasonal Decomposition for {selected_region}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal pattern explanation
            seasonal_pattern = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()
            peak_month = seasonal_pattern.idxmax()
            trough_month = seasonal_pattern.idxmin()
            
            st.markdown(f"""
            <div class="highlight">
                <strong>Seasonal Pattern Insights:</strong><br>
                Peak storage typically occurs in <span class="warning">{month_names[peak_month]}</span> with an average increase of 
                {seasonal_pattern.max():.0f} kilobarrels above the trend.<br>
                Lowest storage typically occurs in <span class="warning">{month_names[trough_month]}</span> with an average decrease of 
                {abs(seasonal_pattern.min()):.0f} kilobarrels below the trend.
            </div>
            """, unsafe_allow_html=True)
            
            # Year-over-Year comparison
            st.markdown('<div class="sub-header">Year-over-Year Comparison</div>', unsafe_allow_html=True)
            
            # Group by year and month
            region_data['Year'] = region_data['ReferenceDate'].dt.year
            yoy_data = region_data.pivot_table(
                values='TotalStorage_KiloBarrels', 
                index='Month', 
                columns='Year',
                aggfunc='mean'
            ).reset_index()
            
            # Add month names
            yoy_data['Month_Name'] = yoy_data['Month'].map(month_names)
            
            # Create figure
            fig = go.Figure()
            
            # Add a line for each year
            for year in yoy_data.columns[1:-1]:  # Skip Month and Month_Name columns
                fig.add_trace(go.Scatter(
                    x=yoy_data['Month_Name'],
                    y=yoy_data[year],
                    mode='lines+markers',
                    name=str(year)
                ))
            
            # Update layout
            fig.update_layout(
                title=f"Year-over-Year Storage Comparison for {selected_region}",
                xaxis_title="Month",
                yaxis_title="Storage (KiloBarrels)",
                xaxis=dict(
                    categoryorder='array',
                    categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                ),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights on YoY changes
            latest_year = region_data['Year'].max()
            previous_year = latest_year - 1
            
            if previous_year in yoy_data.columns:
                # Calculate year-over-year change
                latest_data = yoy_data[latest_year].mean()
                previous_data = yoy_data[previous_year].mean()
                yoy_change = ((latest_data - previous_data) / previous_data) * 100
                
                st.markdown(f"""
                <div class="highlight">
                    <strong>Year-over-Year Insights:</strong><br>
                    Average storage in {latest_year}: <span class="warning">{latest_data:,.0f}</span> kilobarrels<br>
                    Average storage in {previous_year}: <span class="warning">{previous_data:,.0f}</span> kilobarrels<br>
                    Year-over-year change: <span class="{'success' if yoy_change > 0 else 'warning'}">{yoy_change:+.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not perform seasonal analysis: {e}")
    
    # Storage volatility analysis
    st.markdown('<div class="sub-header">Storage Volatility Analysis</div>', unsafe_allow_html=True)
    
    # Calculate rolling volatility (standard deviation)
    volatility_window = 6  # 6-month rolling window
    
    if len(region_data) >= volatility_window + 1:
        # Calculate monthly changes
        region_data['Monthly_Change_Pct'] = region_data['TotalStorage_KiloBarrels'].pct_change() * 100
        
        # Calculate rolling volatility
        region_data['Volatility'] = region_data['Monthly_Change_Pct'].rolling(window=volatility_window).std()
        
        # Filter out NaN values
        volatility_data = region_data.dropna(subset=['Volatility'])
        
        if not volatility_data.empty:
            fig = px.line(
                volatility_data,
                x='ReferenceDate',
                y='Volatility',
                title=f"{volatility_window}-Month Rolling Volatility for {selected_region}",
                markers=True
            )
            
            # Add horizontal line for average volatility
            avg_volatility = volatility_data['Volatility'].mean()
            
            fig.add_shape(
                type="line",
                x0=volatility_data['ReferenceDate'].min(),
                y0=avg_volatility,
                x1=volatility_data['ReferenceDate'].max(),
                y1=avg_volatility,
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                )
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Volatility (Std Dev of Monthly % Changes)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add context to the volatility
            current_vol = volatility_data.iloc[-1]['Volatility']
            vol_comparison = "higher" if current_vol > avg_volatility else "lower"
            vol_pct_diff = abs((current_vol - avg_volatility) / avg_volatility) * 100
            
            st.markdown(f"""
            <div class="highlight">
                <strong>Volatility Insights:</strong><br>
                Current {volatility_window}-month volatility: <span class="warning">{current_vol:.2f}%</span><br>
                Historical average volatility: {avg_volatility:.2f}%<br>
                Current volatility is <span class="{'warning' if vol_comparison == 'higher' else 'success'}">{vol_pct_diff:.1f}% {vol_comparison}</span> than the historical average.
                <br><br>
                <strong>Interpretation:</strong> {'Higher volatility indicates less predictable storage patterns, which may require more conservative planning and larger safety buffers.' if vol_comparison == 'higher' else 'Lower volatility indicates more predictable storage patterns, allowing for more precise capacity planning.'}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"Insufficient data to calculate storage volatility for {selected_region}.")
    else:
        st.info(f"At least {volatility_window + 1} months of data needed for volatility analysis.")

    # Terminal Details Page
elif page == "Terminal Details":
    st.markdown('<div class="main-header">Terminal-Level Analysis</div>', unsafe_allow_html=True)
    
    # Filter terminal data for selected countries
    terminal_data = terminal_level_agg[terminal_level_agg["CountryName"].isin(selected_regions)]
    
    # Filter by date range
    terminal_data = terminal_data[(terminal_data["ReferenceDate"].dt.date >= start_date) & 
                                 (terminal_data["ReferenceDate"].dt.date <= end_date)]
    
    # Get latest date with data
    latest_date = terminal_data["ReferenceDate"].max()
    
    # Terminal selector
    all_terminals = sorted(terminal_data["TerminalName"].unique())
    
    if not all_terminals:
        st.warning("No terminal data available for the selected regions and date range.")
        st.stop()
    
    selected_terminal = st.selectbox("Select a Terminal", all_terminals)
    
    # Filter for selected terminal
    terminal_specific = terminal_data[terminal_data["TerminalName"] == selected_terminal]
    terminal_country = terminal_specific["CountryName"].iloc[0] if not terminal_specific.empty else "Unknown"
    
    # Terminal header and info
    st.markdown(f"""
    <div class="sub-header">Terminal: {selected_terminal}, {terminal_country}</div>
    """, unsafe_allow_html=True)
    
    # Terminal metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Latest storage at this terminal
    latest_terminal = terminal_specific[terminal_specific["ReferenceDate"] == latest_date]
    
    with col1:
        current_storage = latest_terminal["ObservedValue"].iloc[0] if not latest_terminal.empty else 0
        st.metric("Current Storage (KB)", f"{current_storage:,.0f}")
    
    with col2:
        # Calculate average storage
        avg_storage = terminal_specific["ObservedValue"].mean()
        st.metric("Average Storage (KB)", f"{avg_storage:,.0f}")
    
    with col3:
        # Calculate capacity utilization (using max as proxy for capacity)
        max_storage = terminal_specific["ObservedValue"].max()
        utilization = (current_storage / max_storage * 100) if max_storage > 0 and not latest_terminal.empty else 0
        st.metric("Utilization", f"{utilization:.1f}%")
    
    with col4:
        # Calculate MoM change
        if len(terminal_specific) >= 2:
            terminal_specific = terminal_specific.sort_values("ReferenceDate")
            terminal_specific["MoM_Change"] = terminal_specific["ObservedValue"].pct_change() * 100
            latest_change = terminal_specific["MoM_Change"].iloc[-1] if len(terminal_specific) > 1 else 0
            st.metric("MoM Change", f"{latest_change:+.1f}%", delta=f"{latest_change:+.1f}%")
        else:
            st.metric("MoM Change", "N/A")
    
    # Terminal storage trend
    st.markdown('<div class="sub-header">Storage Trend</div>', unsafe_allow_html=True)
    
    fig = px.line(
        terminal_specific,
        x="ReferenceDate",
        y="ObservedValue",
        title=f"Storage Trend for {selected_terminal}",
        markers=True
    )
    
    # Add moving average if enough data
    if len(terminal_specific) >= 3:
        terminal_specific["MA_3M"] = terminal_specific["ObservedValue"].rolling(window=3, min_periods=1).mean()
        
        fig.add_scatter(
            x=terminal_specific["ReferenceDate"],
            y=terminal_specific["MA_3M"],
            mode="lines",
            name="3-Month MA",
            line=dict(color="red", width=2, dash="dash")
        )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Storage (KiloBarrels)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Terminal flow breakdown analysis
    st.markdown('<div class="sub-header">Flow Breakdown Analysis</div>', unsafe_allow_html=True)
    
    # Check if we have flow breakdown data
    flow_breakdown_data = terminals[terminals["TerminalName"] == selected_terminal]
    
    if not flow_breakdown_data.empty:
        # Filter by date range
        flow_breakdown_data = flow_breakdown_data[
            (flow_breakdown_data["ReferenceDate"].dt.date >= start_date) & 
            (flow_breakdown_data["ReferenceDate"].dt.date <= end_date)
        ]
        
        # Get unique flow breakdowns
        flow_types = flow_breakdown_data["FlowBreakdown"].unique()
        
        if len(flow_types) > 1:  # If we have multiple flow types
            # Create a pivot table for visualization
            flow_pivot = flow_breakdown_data.pivot_table(
                index="ReferenceDate", 
                columns="FlowBreakdown", 
                values="ObservedValue",
                aggfunc="sum"
            ).reset_index()
            
            # Convert to long format for plotly
            flow_long = pd.melt(
                flow_pivot, 
                id_vars=["ReferenceDate"],
                value_vars=[col for col in flow_pivot.columns if col != "ReferenceDate"],
                var_name="Flow Type",
                value_name="Volume"
            )
            
            # Create stacked area chart
            fig = px.area(
                flow_long,
                x="ReferenceDate",
                y="Volume",
                color="Flow Type",
                title=f"Storage Composition for {selected_terminal}"
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Storage Volume (KiloBarrels)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Current composition pie chart
            latest_flow = flow_breakdown_data[flow_breakdown_data["ReferenceDate"] == flow_breakdown_data["ReferenceDate"].max()]
            
            if not latest_flow.empty:
                fig = px.pie(
                    latest_flow,
                    values="ObservedValue",
                    names="FlowBreakdown",
                    title=f"Current Storage Composition ({latest_flow['ReferenceDate'].iloc[0].strftime('%b %Y')})"
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Only one flow type available for this terminal.")
    else:
        st.info("No detailed flow breakdown data available for this terminal.")
    
    # Terminal comparison within country
    st.markdown('<div class="sub-header">Terminal Comparison within Country</div>', unsafe_allow_html=True)
    
    # Get all terminals in the same country
    country_terminals = terminal_data[terminal_data["CountryName"] == terminal_country]
    
    # Get latest data for comparison
    latest_country_terminals = country_terminals[country_terminals["ReferenceDate"] == latest_date]
    
    if not latest_country_terminals.empty:
        # Group by terminal
        latest_terminal_totals = latest_country_terminals.groupby("TerminalName")["ObservedValue"].sum().reset_index()
        
        # Sort by storage volume
        latest_terminal_totals = latest_terminal_totals.sort_values("ObservedValue", ascending=False)
        
        fig = px.bar(
            latest_terminal_totals,
            x="TerminalName",
            y="ObservedValue",
            title=f"Terminal Comparison in {terminal_country} ({latest_date.strftime('%b %Y')})",
            color="ObservedValue",
            color_continuous_scale="Viridis"
        )
        
        # Highlight selected terminal
        selected_data = latest_terminal_totals[latest_terminal_totals["TerminalName"] == selected_terminal]
        if not selected_data.empty:
            # Get the x-position based on the terminal's position in the bar chart
            terminal_position = latest_terminal_totals["TerminalName"].tolist().index(selected_terminal)
            terminal_value = selected_data["ObservedValue"].values[0]
            
            fig.add_shape(
                type="rect",
                x0=terminal_position - 0.4,
                y0=0,
                x1=terminal_position + 0.4,
                y1=terminal_value,
                line=dict(color="red", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
        
        fig.update_layout(
            xaxis_title="Terminal",
            yaxis_title="Storage (KiloBarrels)",
            height=400
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate market share
        total_country_storage = latest_terminal_totals["ObservedValue"].sum()
        terminal_share = (latest_terminal_totals[latest_terminal_totals["TerminalName"] == selected_terminal]["ObservedValue"].iloc[0] / total_country_storage * 100) if not latest_terminal_totals[latest_terminal_totals["TerminalName"] == selected_terminal].empty else 0
        
        st.markdown(f"""
        <div class="highlight">
            <strong>Market Position Insights:</strong><br>
            {selected_terminal} represents <span class="warning">{terminal_share:.1f}%</span> of total storage capacity in {terminal_country}.<br>
            Ranking: {(latest_terminal_totals["TerminalName"] == selected_terminal).idxmax() + 1} out of {len(latest_terminal_totals)} terminals in the country.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"No recent data available for terminals in {terminal_country}.")
    
    # Weekly storage patterns (if available)
    st.markdown('<div class="sub-header">Weekly Storage Patterns</div>', unsafe_allow_html=True)
    
    # Check if we have weekly data
    weekly_data = terminals_weekly[terminals_weekly["TerminalName"] == selected_terminal]
    
    if not weekly_data.empty:
        # Filter by date range
        weekly_data = weekly_data[
            (weekly_data["ReferenceDate"].dt.date >= start_date) & 
            (weekly_data["ReferenceDate"].dt.date <= end_date)
        ]
        
        if not weekly_data.empty:
            # Create a line chart for weekly patterns
            fig = px.line(
                weekly_data,
                x="ReferenceDate",
                y="ObservedValue",
                title=f"Weekly Storage Pattern for {selected_terminal}",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Storage (KiloBarrels)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly volatility
            if len(weekly_data) > 1:
                weekly_data["WoW_Change"] = weekly_data["ObservedValue"].pct_change() * 100
                weekly_volatility = weekly_data["WoW_Change"].std()
                
                st.markdown(f"""
                <div class="highlight">
                    <strong>Weekly Volatility:</strong><br>
                    Standard deviation of week-over-week changes: <span class="warning">{weekly_volatility:.2f}%</span><br>
                    {'<span class="warning">High weekly volatility</span> indicates significant operational fluctuations.' if weekly_volatility > 5 else '<span class="success">Low weekly volatility</span> suggests stable operations.'}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No weekly data available within the selected date range.")
    else:
        st.info("No weekly data available for this terminal.")
    
    # Storage utilization forecast
    if len(terminal_specific) >= 12:  # Only if we have enough historical data
        st.markdown('<div class="sub-header">Storage Utilization Forecast</div>', unsafe_allow_html=True)
        
        # Simple time series forecast for next 3 months
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Prepare time series data
        ts_data = terminal_specific.set_index("ReferenceDate")["ObservedValue"]
        
        try:
            # Fit model
            model = ExponentialSmoothing(
                ts_data,
                trend='add',
                seasonal='add',
                seasonal_periods=12
            )
            model_fit = model.fit()
            
            # Forecast next 3 months
            forecast_horizon = 3
            forecast_values = model_fit.forecast(forecast_horizon)
            
            # Create forecast index
            forecast_index = pd.date_range(start=ts_data.index[-1], periods=forecast_horizon + 1, freq="MS")[1:]
            
            # Plot
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=forecast_values.values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', dash='dash')
            ))
            
            # Add capacity line (using max as proxy)
            capacity = ts_data.max() * 1.1  # Add 10% buffer
            
            fig.add_shape(
                type="line",
                x0=ts_data.index.min(),
                y0=capacity,
                x1=forecast_index[-1],
                y1=capacity,
                line=dict(
                    color="green",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add text annotation for capacity
            fig.add_annotation(
                x=forecast_index[-1],
                y=capacity,
                text="Estimated Capacity",
                showarrow=False,
                yshift=10,
                font=dict(
                    color="green"
                )
            )
            
            # Update layout
            fig.update_layout(
                title=f"3-Month Storage Forecast for {selected_terminal}",
                xaxis_title="Date",
                yaxis_title="Storage (KiloBarrels)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate utilization forecast
            max_forecast = forecast_values.max()
            utilization_forecast = (max_forecast / capacity) * 100
            
            status = "Critical" if utilization_forecast > 90 else "Warning" if utilization_forecast > 80 else "Normal"
            status_color = "warning" if status in ["Critical", "Warning"] else "success"
            
            st.markdown(f"""
            <div class="highlight">
                <strong>Utilization Forecast:</strong><br>
                Maximum forecasted storage: <span class="warning">{max_forecast:,.0f}</span> kilobarrels<br>
                Estimated capacity: {capacity:,.0f} kilobarrels<br>
                Maximum utilization forecast: <span class="{status_color}">{utilization_forecast:.1f}%</span> ({status})
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not generate forecast: {e}")

# Floating Storage Page
elif page == "Floating Storage":
    st.markdown('<div class="main-header">Floating Storage Analysis</div>', unsafe_allow_html=True)
    
    # Filter floating storage data
    filtered_floating = floating_storage[
        (floating_storage["ReferenceDate"].dt.date >= start_date) & 
        (floating_storage["ReferenceDate"].dt.date <= end_date)
    ]
    
    # Filter for selected regions if applicable
    if 'OriginCountry' in filtered_floating.columns:
        region_filtered_floating = filtered_floating[filtered_floating["OriginCountry"].isin(selected_regions)]
    else:
        region_filtered_floating = filtered_floating
        st.warning("No region information available in floating storage data. Showing all data.")
    
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_floating = region_filtered_floating["QuantityKiloBarrels"].sum()
        st.metric("Total Floating Storage (KB)", f"{total_floating:,.0f}")
    
    with col2:
        vessel_count = region_filtered_floating["IMO"].nunique() if "IMO" in region_filtered_floating.columns else "N/A"
        st.metric("Active Vessels", f"{vessel_count:,}")
    
    with col3:
        if "StartDate" in region_filtered_floating.columns and "EndDate" in region_filtered_floating.columns:
            # Calculate average storage duration
            region_filtered_floating["Duration"] = (region_filtered_floating["EndDate"] - region_filtered_floating["StartDate"]).dt.days
            avg_duration = region_filtered_floating["Duration"].mean()
            st.metric("Avg. Storage Duration (days)", f"{avg_duration:.0f}")
        else:
            st.metric("Avg. Storage Duration", "N/A")
    
    with col4:
        # Latest date
        latest_date = region_filtered_floating["ReferenceDate"].max()
        # Month-over-month change
        if "ReferenceDate" in region_filtered_floating.columns:
            date_grouped = region_filtered_floating.groupby("ReferenceDate")["QuantityKiloBarrels"].sum().reset_index()
            if len(date_grouped) >= 2:
                date_grouped = date_grouped.sort_values("ReferenceDate")
                date_grouped["MoM_Change"] = date_grouped["QuantityKiloBarrels"].pct_change() * 100
                latest_change = date_grouped["MoM_Change"].iloc[-1]
                st.metric("MoM Change", f"{latest_change:+.1f}%", delta=f"{latest_change:+.1f}%")
            else:
                st.metric("MoM Change", "N/A")
        else:
            st.metric("MoM Change", "N/A")
    
    # Tabs for different floating storage analyses
    tabs = st.tabs(["Geographic Distribution", "Vessel Analysis", "Time Series Analysis", "Vessel Movement"])
    
    # Tab 1: Geographic Distribution
    with tabs[0]:
        st.markdown('<div class="sub-header">Geographic Distribution of Floating Storage</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Area distribution
            if "Area" in region_filtered_floating.columns:
                area_storage = region_filtered_floating.groupby("Area")["QuantityKiloBarrels"].sum().reset_index()
                area_storage = area_storage.sort_values("QuantityKiloBarrels", ascending=False)
                
                fig = px.pie(
                    area_storage,
                    values="QuantityKiloBarrels",
                    names="Area",
                    title="Storage by Geographic Area",
                    hole=0.4
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No area information available in the data.")
        
        with col2:
            # Country distribution
            if "OriginCountry" in region_filtered_floating.columns:
                country_storage = region_filtered_floating.groupby("OriginCountry")["QuantityKiloBarrels"].sum().reset_index()
                country_storage = country_storage.sort_values("QuantityKiloBarrels", ascending=False)
                
                fig = px.pie(
                    country_storage,
                    values="QuantityKiloBarrels",
                    names="OriginCountry",
                    title="Storage by Origin Country",
                    hole=0.4
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No country information available in the data.")
        
        # Area trends over time
        if "Area" in region_filtered_floating.columns and "ReferenceDate" in region_filtered_floating.columns:
            st.markdown('<div class="sub-header">Area Trends Over Time</div>', unsafe_allow_html=True)
            
            # Group by area and date
            area_time = region_filtered_floating.groupby(["Area", "ReferenceDate"])["QuantityKiloBarrels"].sum().reset_index()
            
            # Get top 5 areas by volume
            top_areas = area_storage.head(5)["Area"].tolist()
            
            # Filter for top areas
            top_area_data = area_time[area_time["Area"].isin(top_areas)]
            
            # Create time series visualization
            fig = px.line(
                top_area_data,
                x="ReferenceDate",
                y="QuantityKiloBarrels",
                color="Area",
                title="Floating Storage Trends by Major Areas",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Storage (KiloBarrels)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Heatmap of area storage over time
            st.markdown('<div class="sub-header">Storage Heatmap by Area</div>', unsafe_allow_html=True)
            
            # Create pivot table for heatmap
            heatmap_data = area_time.pivot_table(
                index="ReferenceDate",
                columns="Area",
                values="QuantityKiloBarrels",
                aggfunc="sum"
            ).fillna(0)
            
            # Create heatmap
            fig = px.imshow(
                heatmap_data,
                title="Storage Volume by Area Over Time",
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(
                xaxis_title="Geographic Area",
                yaxis_title="Date",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        if "Area" in region_filtered_floating.columns:
            # Get latest data
            latest_date = region_filtered_floating["ReferenceDate"].max()
            latest_data = region_filtered_floating[region_filtered_floating["ReferenceDate"] == latest_date]
            
            # Identify areas with substantial floating storage
            top_current_areas = latest_data.groupby("Area")["QuantityKiloBarrels"].sum().reset_index()
            top_current_areas = top_current_areas.sort_values("QuantityKiloBarrels", ascending=False)
            
            st.markdown(f"""
            <div class="highlight">
                <strong>Geographic Insights:</strong><br>
                Top storage area: <span class="warning">{top_current_areas.iloc[0]['Area']}</span> with {top_current_areas.iloc[0]['QuantityKiloBarrels']:,.0f} kilobarrels<br>
                {top_current_areas.iloc[0]['Area']} represents {top_current_areas.iloc[0]['QuantityKiloBarrels']/total_floating*100:.1f}% of total floating storage.<br>
                Top 3 areas account for {top_current_areas.head(3)['QuantityKiloBarrels'].sum()/total_floating*100:.1f}% of all floating storage.
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 2: Vessel Analysis
    with tabs[1]:
        st.markdown('<div class="sub-header">Vessel Class Analysis</div>', unsafe_allow_html=True)
        
        if "VesselClass" in region_filtered_floating.columns:
            # Vessel class distribution
            vessel_class_storage = region_filtered_floating.groupby("VesselClass")["QuantityKiloBarrels"].sum().reset_index()
            vessel_class_storage = vessel_class_storage.sort_values("QuantityKiloBarrels", ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Storage by vessel class
                fig = px.bar(
                    vessel_class_storage,
                    x="VesselClass",
                    y="QuantityKiloBarrels",
                    color="VesselClass",
                    title="Storage by Vessel Class"
                )
                
                fig.update_layout(
                    xaxis_title="Vessel Class",
                    yaxis_title="Storage (KiloBarrels)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Vessel count by class
                if "IMO" in region_filtered_floating.columns:
                    vessel_counts = region_filtered_floating.groupby("VesselClass")["IMO"].nunique().reset_index()
                    vessel_counts.columns = ["VesselClass", "VesselCount"]
                    vessel_counts = vessel_counts.sort_values("VesselCount", ascending=False)
                    
                    fig = px.bar(
                        vessel_counts,
                        x="VesselClass",
                        y="VesselCount",
                        color="VesselClass",
                        title="Number of Vessels by Class"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Vessel Class",
                        yaxis_title="Vessel Count",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No vessel identification information available.")
            
            # Average storage by vessel class
            if "IMO" in region_filtered_floating.columns:
                avg_storage_by_class = region_filtered_floating.groupby(["VesselClass", "IMO"])["QuantityKiloBarrels"].mean().reset_index()
                avg_by_class = avg_storage_by_class.groupby("VesselClass")["QuantityKiloBarrels"].mean().reset_index()
                avg_by_class.columns = ["VesselClass", "AvgStorage"]
                avg_by_class = avg_by_class.sort_values("AvgStorage", ascending=False)
                
                fig = px.bar(
                    avg_by_class,
                    x="VesselClass",
                    y="AvgStorage",
                    color="VesselClass",
                    title="Average Storage per Vessel by Class"
                )
                
                fig.update_layout(
                    xaxis_title="Vessel Class",
                    yaxis_title="Average Storage (KiloBarrels)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Top vessels table
            if "VesselName" in region_filtered_floating.columns and "IMO" in region_filtered_floating.columns:
                st.markdown('<div class="sub-header">Top Vessels by Storage Volume</div>', unsafe_allow_html=True)
                
                # Group by vessel and take most recent data
                latest_vessels = region_filtered_floating.sort_values("ReferenceDate", ascending=False)
                latest_vessels = latest_vessels.drop_duplicates(subset=["IMO"])
                
                # Sort by storage volume
                top_vessels = latest_vessels.sort_values("QuantityKiloBarrels", ascending=False).head(10)
                
                # Format for display
                display_cols = ["VesselName", "IMO", "VesselClass", "QuantityKiloBarrels", "Area", "OriginCountry"]
                display_cols = [col for col in display_cols if col in top_vessels.columns]
                
                if display_cols:
                    top_vessels_display = top_vessels[display_cols]
                    rename_dict = {
                        "VesselName": "Vessel Name",
                        "IMO": "IMO Number",
                        "VesselClass": "Class",
                        "QuantityKiloBarrels": "Storage (KB)",
                        "Area": "Area",
                        "OriginCountry": "Origin"
                    }
                    
                    # Only include columns that exist
                    rename_dict = {k: v for k, v in rename_dict.items() if k in top_vessels_display.columns}
                    
                    top_vessels_display = top_vessels_display.rename(columns=rename_dict)
                    
                    st.dataframe(
                        top_vessels_display.set_index("Vessel Name" if "VesselName" in top_vessels.columns else "IMO Number")
                        .style.format({
                            "Storage (KB)": "{:,.0f}"
                        })
                        .background_gradient(cmap="viridis", subset=["Storage (KB)"])
                    )
        else:
            st.info("No vessel class information available in the data.")
        
        # Vessel class trends over time
        if "VesselClass" in region_filtered_floating.columns and "ReferenceDate" in region_filtered_floating.columns:
            st.markdown('<div class="sub-header">Vessel Class Trends Over Time</div>', unsafe_allow_html=True)
            
            # Group by vessel class and date
            class_time = region_filtered_floating.groupby(["VesselClass", "ReferenceDate"])["QuantityKiloBarrels"].sum().reset_index()
            
            # Create time series visualization
            fig = px.line(
                class_time,
                x="ReferenceDate",
                y="QuantityKiloBarrels",
                color="VesselClass",
                title="Floating Storage Trends by Vessel Class",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Storage (KiloBarrels)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights on vessel class trends
            if len(class_time) > 0:
                # Calculate growth rates for each vessel class
                growth_rates = []
                
                for vessel_class in class_time["VesselClass"].unique():
                    class_data = class_time[class_time["VesselClass"] == vessel_class].sort_values("ReferenceDate")
                    
                    if len(class_data) >= 2:
                        first_value = class_data["QuantityKiloBarrels"].iloc[0]
                        last_value = class_data["QuantityKiloBarrels"].iloc[-1]
                        
                        if first_value > 0:
                            growth_rate = (last_value - first_value) / first_value * 100
                            growth_rates.append({
                                "VesselClass": vessel_class,
                                "GrowthRate": growth_rate,
                                "StartVolume": first_value,
                                "EndVolume": last_value
                            })
                
                if growth_rates:
                    growth_df = pd.DataFrame(growth_rates)
                    growth_df = growth_df.sort_values("GrowthRate", ascending=False)
                    
                    # Create growth rate visualization
                    fig = px.bar(
                        growth_df,
                        x="VesselClass",
                        y="GrowthRate",
                        color="VesselClass",
                        title="Growth Rate by Vessel Class",
                        text="GrowthRate"
                    )
                    
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    
                    fig.update_layout(
                        xaxis_title="Vessel Class",
                        yaxis_title="Growth Rate (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add insights
                    fastest_growing = growth_df.iloc[0]
                    fastest_declining = growth_df.iloc[-1] if growth_df.iloc[-1]["GrowthRate"] < 0 else None
                    
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Vessel Class Trend Insights:</strong><br>
                        Fastest growing: <span class="success">{fastest_growing['VesselClass']}</span> with {fastest_growing['GrowthRate']:.1f}% growth<br>
                        {f"Fastest declining: <span class='warning'>{fastest_declining['VesselClass']}</span> with {fastest_declining['GrowthRate']:.1f}% decline<br>" if fastest_declining is not None else ""}
                        This suggests {'increasing demand for larger vessels' if 'VLCC' in fastest_growing['VesselClass'] or 'Suezmax' in fastest_growing['VesselClass'] else 'changing trade patterns requiring different vessel types'}.
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 3: Time Series Analysis
    with tabs[2]:
        st.markdown('<div class="sub-header">Floating Storage Time Series Analysis</div>', unsafe_allow_html=True)
        
        # Aggregate by date
        if "ReferenceDate" in region_filtered_floating.columns:
            time_series = region_filtered_floating.groupby("ReferenceDate")["QuantityKiloBarrels"].sum().reset_index()
            time_series = time_series.sort_values("ReferenceDate")
            
            # Create time series plot
            fig = px.line(
                time_series,
                x="ReferenceDate",
                y="QuantityKiloBarrels",
                title="Total Floating Storage Over Time",
                markers=True
            )
            
            # Add trend line
            if len(time_series) >= 3:
                time_series["MA_3M"] = time_series["QuantityKiloBarrels"].rolling(window=3, min_periods=1).mean()
                
                fig.add_scatter(
                    x=time_series["ReferenceDate"],
                    y=time_series["MA_3M"],
                    mode="lines",
                    name="3-Period Moving Avg",
                    line=dict(color="red", width=2, dash="dash")
                )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Total Storage (KiloBarrels)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Monthly changes
            if len(time_series) >= 2:
                time_series["MoM_Change"] = time_series["QuantityKiloBarrels"].pct_change() * 100
                time_series["MoM_Change_Abs"] = time_series["QuantityKiloBarrels"].diff()
                
                fig = px.bar(
                    time_series.dropna(),
                    x="ReferenceDate",
                    y="MoM_Change_Abs",
                    title="Monthly Change in Floating Storage",
                    color=np.where(time_series.dropna()["MoM_Change_Abs"] >= 0, "Increase", "Decrease"),
                    color_discrete_map={"Increase": "#059669", "Decrease": "#DC2626"}
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Change (KiloBarrels)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal patterns if enough data
            if len(time_series) >= 12:
                st.markdown('<div class="sub-header">Seasonal Patterns in Floating Storage</div>', unsafe_allow_html=True)
                
                # Extract month and calculate monthly averages
                time_series["Month"] = time_series["ReferenceDate"].dt.month
                
                seasonal_data = time_series.groupby("Month")["QuantityKiloBarrels"].mean().reset_index()
                
                # Map month numbers to names
                month_names = {
                    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
                }
                
                seasonal_data["Month_Name"] = seasonal_data["Month"].map(month_names)
                
                fig = px.line(
                    seasonal_data,
                    x="Month_Name",
                    y="QuantityKiloBarrels",
                    markers=True,
                    title="Average Floating Storage by Month"
                )
                
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Average Storage (KiloBarrels)",
                    height=400,
                    xaxis=dict(
                        categoryorder='array',
                        categoryarray=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add seasonal insights
                peak_month = seasonal_data.loc[seasonal_data["QuantityKiloBarrels"].idxmax()]
                low_month = seasonal_data.loc[seasonal_data["QuantityKiloBarrels"].idxmin()]
                
                st.markdown(f"""
                <div class="highlight">
                    <strong>Seasonal Pattern Insights:</strong><br>
                    Peak floating storage typically occurs in <span class="warning">{peak_month['Month_Name']}</span> 
                    with an average of {peak_month['QuantityKiloBarrels']:,.0f} kilobarrels.<br>
                    Lowest floating storage typically occurs in <span class="warning">{low_month['Month_Name']}</span> 
                    with an average of {low_month['QuantityKiloBarrels']:,.0f} kilobarrels.<br>
                    This seasonal pattern may correlate with {'refinery maintenance schedules' if peak_month['Month'] in [3, 4, 9, 10] else 'seasonal demand fluctuations'}.
                </div>
                """, unsafe_allow_html=True)
            
            # Correlation with other factors
            st.markdown('<div class="sub-header">Correlation Analysis</div>', unsafe_allow_html=True)
            
            # Generate random correlation data for demonstration (replace with actual data in production)
            # In a real implementation, you would merge with price data or other factors
            
            # Simulate correlation with price
            np.random.seed(42)  # For reproducibility
            time_series["Price"] = np.random.normal(60, 10, size=len(time_series))
            time_series["Price"] = time_series["Price"].cumsum() / 10
            
            # Create scatter plot
            fig = px.scatter(
                time_series,
                x="Price",
                y="QuantityKiloBarrels",
                title="Floating Storage vs Price",
                trendline="ols"
            )
            
            fig.update_layout(
                xaxis_title="Price ($/bbl)",
                yaxis_title="Floating Storage (KiloBarrels)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation
            correlation = time_series["QuantityKiloBarrels"].corr(time_series["Price"])
            
            st.markdown(f"""
            <div class="highlight">
                <strong>Correlation Analysis:</strong><br>
                Correlation coefficient between price and floating storage: <span class="{'warning' if correlation < 0 else 'success'}">{correlation:.2f}</span><br>
                This suggests a {'negative' if correlation < 0 else 'positive'} relationship between price and floating storage volumes.
                {'Higher prices tend to reduce floating storage as it becomes economical to sell inventory.' if correlation < 0 else 
                'Higher prices may encourage more storage in anticipation of further price increases.'}
            </div>
            """, unsafe_allow_html=True)
    
    # Tab 4: Vessel Movement
    with tabs[3]:
        st.markdown('<div class="sub-header">Vessel Movement Analysis</div>', unsafe_allow_html=True)
        
        if "StartDate" in region_filtered_floating.columns and "EndDate" in region_filtered_floating.columns:
            # Calculate storage duration
            region_filtered_floating["Duration"] = (region_filtered_floating["EndDate"] - region_filtered_floating["StartDate"]).dt.days
            
            # Duration distribution
            fig = px.histogram(
                region_filtered_floating,
                x="Duration",
                nbins=20,
                title="Distribution of Storage Duration",
                color_discrete_sequence=["blue"]
            )
            
            fig.update_layout(
                xaxis_title="Duration (Days)",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration by vessel class if available
            if "VesselClass" in region_filtered_floating.columns:
                duration_by_class = region_filtered_floating.groupby("VesselClass")["Duration"].mean().reset_index()
                duration_by_class = duration_by_class.sort_values("Duration", ascending=False)
                
                fig = px.bar(
                    duration_by_class,
                    x="VesselClass",
                    y="Duration",
                    color="VesselClass",
                    title="Average Storage Duration by Vessel Class"
                )
                
                fig.update_layout(
                    xaxis_title="Vessel Class",
                    yaxis_title="Average Duration (Days)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Duration by area if available
            if "Area" in region_filtered_floating.columns:
                duration_by_area = region_filtered_floating.groupby("Area")["Duration"].mean().reset_index()
                duration_by_area = duration_by_area.sort_values("Duration", ascending=False)
                
                fig = px.bar(
                    duration_by_area,
                    x="Area",
                    y="Duration",
                    color="Area",
                    title="Average Storage Duration by Area"
                )
                
                fig.update_layout(
                    xaxis_title="Area",
                    yaxis_title="Average Duration (Days)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Long-term storage vessels
            long_term_threshold = 30  # 30 days or more
            long_term_vessels = region_filtered_floating[region_filtered_floating["Duration"] >= long_term_threshold]
            
            if not long_term_vessels.empty and "VesselName" in long_term_vessels.columns:
                st.markdown('<div class="sub-header">Long-Term Storage Vessels</div>', unsafe_allow_html=True)
                
                # Group by vessel
                vessel_durations = long_term_vessels.groupby(["VesselName", "IMO" if "IMO" in long_term_vessels.columns else "VesselClass"])["Duration"].max().reset_index()
                vessel_durations = vessel_durations.sort_values("Duration", ascending=False).head(10)
                
                # Format for display
                display_cols = ["VesselName", "IMO" if "IMO" in vessel_durations.columns else "VesselClass", "Duration"]
                display_cols = [col for col in display_cols if col in vessel_durations.columns]
                
                if display_cols:
                    long_term_display = vessel_durations[display_cols]
                    rename_dict = {
                        "VesselName": "Vessel Name",
                        "IMO": "IMO Number",
                        "VesselClass": "Class",
                        "Duration": "Days in Storage"
                    }
                    
                    # Only include columns that exist
                    rename_dict = {k: v for k, v in rename_dict.items() if k in long_term_display.columns}
                    
                    long_term_display = long_term_display.rename(columns=rename_dict)
                    
                    st.dataframe(
                        long_term_display.set_index("Vessel Name" if "VesselName" in vessel_durations.columns else "Class")
                        .style.format({
                            "Days in Storage": "{:.0f}"
                        })
                        .background_gradient(cmap="viridis", subset=["Days in Storage"])
                    )
                    
                    # Add insights
                    max_duration = vessel_durations["Duration"].max()
                    min_duration = vessel_durations["Duration"].min()
                    avg_duration = vessel_durations["Duration"].mean()
                    
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Long-Term Storage Insights:</strong><br>
                        Longest storage duration: <span class="warning">{max_duration:.0f}</span> days<br>
                        Average long-term storage duration: {avg_duration:.0f} days<br>
                        {len(long_term_vessels)} vessels have been in storage for {long_term_threshold}+ days.<br>
                        Long-term storage may indicate {'market contango' if len(long_term_vessels) > 10 else 'limited shore tank availability or trading strategy'}.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No vessel movement data (start/end dates) available for analysis.")
        
        # Storage event count over time if available
        if "StartDate" in region_filtered_floating.columns:
            st.markdown('<div class="sub-header">Storage Events Over Time</div>', unsafe_allow_html=True)
            
            # Count storage starts by date
            region_filtered_floating["StartDate_Day"] = region_filtered_floating["StartDate"].dt.floor("D")
            start_counts = region_filtered_floating.groupby("StartDate_Day").size().reset_index()
            start_counts.columns = ["Date", "NewStorageEvents"]
            
            # Count storage ends by date if available
            if "EndDate" in region_filtered_floating.columns:
                region_filtered_floating["EndDate_Day"] = region_filtered_floating["EndDate"].dt.floor("D")
                end_counts = region_filtered_floating.groupby("EndDate_Day").size().reset_index()
                end_counts.columns = ["Date", "StorageEndEvents"]
                
                # Merge start and end events
                event_counts = pd.merge(start_counts, end_counts, on="Date", how="outer").fillna(0)
            else:
                event_counts = start_counts
                event_counts["StorageEndEvents"] = 0
            
            # Sort by date
            event_counts = event_counts.sort_values("Date")
            
            # Calculate net change
            event_counts["NetChange"] = event_counts["NewStorageEvents"] - event_counts["StorageEndEvents"]
            
            # Create visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=event_counts["Date"],
                y=event_counts["NewStorageEvents"],
                name="New Storage",
                marker_color="green"
            ))
            
            fig.add_trace(go.Bar(
                x=event_counts["Date"],
                y=-event_counts["StorageEndEvents"],
                name="Storage Ended",
                marker_color="red"
            ))
            
            fig.add_trace(go.Scatter(
                x=event_counts["Date"],
                y=event_counts["NetChange"],
                name="Net Change",
                line=dict(color="blue", width=2)
            ))
            
            fig.update_layout(
                title="Floating Storage Event Timeline",
                xaxis_title="Date",
                yaxis_title="Event Count",
                barmode="relative",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights
            recent_period = event_counts.tail(30)  # Last 30 days with data
            net_change = recent_period["NetChange"].sum()
            
            st.markdown(f"""
            <div class="highlight">
                <strong>Recent Storage Activity Insights:</strong><br>
                Net change in storage events (last 30 data points): <span class="{'success' if net_change > 0 else 'warning'}">{net_change:+.0f}</span> vessels<br>
                New storage events: {recent_period["NewStorageEvents"].sum():.0f}<br>
                Storage end events: {recent_period["StorageEndEvents"].sum():.0f}<br>
                This suggests {'increasing floating storage activity' if net_change > 0 else 'decreasing floating storage activity'}.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No storage event timing data available for analysis.")

# Forecasting Page
elif page == "Forecasting":
    st.markdown('<div class="main-header">Storage Forecasting</div>', unsafe_allow_html=True)
    
    # Region selector
    selected_region = st.selectbox("Select a Region for Forecasting", selected_regions)
    
    # Filter data for selected region
    region_data = filtered_data[filtered_data["CountryName"] == selected_region]
    
    if region_data.empty:
        st.warning(f"No data available for {selected_region}.")
        st.stop()
    
    # Forecast parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_horizon = st.slider("Forecast Horizon (Months)", min_value=3, max_value=24, value=12)
    
    with col2:
        confidence_level = st.slider("Confidence Level (%)", min_value=80, max_value=99, value=95)
    
    with col3:
        model_type = st.selectbox("Model Type", ["SARIMA", "Exponential Smoothing", "Moving Average"])
    
    # Prepare data for forecasting
    forecast_data = region_data.set_index("ReferenceDate")["TotalStorage_KiloBarrels"].dropna()
    
    if len(forecast_data) < 12:
        st.warning("Insufficient data for reliable forecasting (need at least 12 months).")
        st.stop()
    
    # Additional model-specific parameters
    if model_type == "SARIMA":
        st.markdown('<div class="sub-header">SARIMA Model Parameters</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p = st.slider("AR Order (p)", 0, 3, 1)
            P = st.slider("Seasonal AR Order (P)", 0, 2, 1)
        
        with col2:
            d = st.slider("Differencing (d)", 0, 2, 1)
            D = st.slider("Seasonal Differencing (D)", 0, 1, 1)
        
        with col3:
            q = st.slider("MA Order (q)", 0, 3, 1)
            Q = st.slider("Seasonal MA Order (Q)", 0, 2, 1)
    
    elif model_type == "Exponential Smoothing":
        st.markdown('<div class="sub-header">Exponential Smoothing Parameters</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trend_type = st.selectbox("Trend Type", ["additive", "multiplicative", None], index=0)
        
        with col2:
            seasonal_type = st.selectbox("Seasonal Type", ["additive", "multiplicative", None], index=0)
        
        with col3:
            seasonal_periods = st.slider("Seasonal Periods", min_value=4, max_value=12, value=12)
    
    elif model_type == "Moving Average":
        st.markdown('<div class="sub-header">Moving Average Parameters</div>', unsafe_allow_html=True)
        
        ma_window = st.slider("Moving Average Window", min_value=2, max_value=12, value=3)
    
    # Generate forecast
    st.markdown('<div class="sub-header">Storage Forecast</div>', unsafe_allow_html=True)
    
    # Create placeholder for forecast results
    forecast_placeholder = st.empty()
    
    with st.spinner("Generating forecast..."):
        try:
            if model_type == "SARIMA":
                # SARIMA model
                model = SARIMAX(
                    forecast_data, 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_fit = model.fit(disp=False)
                
                # Generate forecast
                forecast_result = model_fit.get_forecast(steps=forecast_horizon)
                forecast_mean = forecast_result.predicted_mean
                
                # Get confidence intervals
                alpha = 1 - (confidence_level / 100)
                forecast_ci = forecast_result.conf_int(alpha=alpha)
                
                # Create forecast index
                forecast_index = pd.date_range(start=forecast_data.index[-1], periods=forecast_horizon + 1, freq="MS")[1:]
                
                # Create figure
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast_mean.values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                    y=forecast_ci.iloc[:, 0].tolist() + forecast_ci.iloc[:, 1].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(231,107,243,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Storage Forecast for {selected_region} using SARIMA({p},{d},{q})({P},{D},{Q})12",
                    xaxis_title="Date",
                    yaxis_title="Storage (KiloBarrels)",
                    height=500
                )
                
                forecast_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Model diagnostics
                st.markdown('<div class="sub-header">Model Diagnostics</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # AIC and BIC
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Model Fit Statistics:</strong><br>
                        AIC: {model_fit.aic:.2f}<br>
                        BIC: {model_fit.bic:.2f}<br>
                        Log Likelihood: {model_fit.llf:.2f}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Ljung-Box test for autocorrelation
                    ljung_box = sm.stats.acorr_ljungbox(model_fit.resid, lags=[10], return_df=True)
                    p_value = ljung_box["lb_pvalue"].iloc[0]
                    
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Residual Diagnostics:</strong><br>
                        Ljung-Box Test p-value: {p_value:.4f}<br>
                        {'<span class="success">No significant autocorrelation in residuals</span>' if p_value > 0.05 else '<span class="warning">Possible autocorrelation in residuals</span>'}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Residual plots
                with st.expander("View Residual Plots"):
                    # Create residual plots
                    fig = make_subplots(rows=2, cols=2, 
                                      subplot_titles=("Residuals", "Residual Distribution", 
                                                      "Residual ACF", "Residual PACF"))
                    
                    # Residuals vs time
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index,
                        y=model_fit.resid,
                        mode='lines',
                        name='Residuals'
                    ), row=1, col=1)
                    
                    # Add zero line
                    fig.add_shape(
                        type="line",
                        x0=forecast_data.index.min(),
                        y0=0,
                        x1=forecast_data.index.max(),
                        y1=0,
                        line=dict(color="red", dash="dash"),
                        row=1, col=1
                    )
                    
                    # Residual distribution
                    fig.add_trace(go.Histogram(
                        x=model_fit.resid,
                        nbinsx=20,
                        name='Residual Distribution'
                    ), row=1, col=2)
                    
                    # ACF and PACF
                    acf_values = sm.tsa.acf(model_fit.resid, nlags=20)
                    pacf_values = sm.tsa.pacf(model_fit.resid, nlags=20)
                    lags = list(range(len(acf_values)))
                    
                    fig.add_trace(go.Bar(
                        x=lags,
                        y=acf_values,
                        name='ACF'
                    ), row=2, col=1)
                    
                    fig.add_trace(go.Bar(
                        x=lags,
                        y=pacf_values,
                        name='PACF'
                    ), row=2, col=2)
                    
                    # Add confidence bands to ACF/PACF
                    conf_level = 1.96 / np.sqrt(len(forecast_data))
                    
                    for i in range(2):
                        for j in range(1, 3):
                            fig.add_shape(
                                type="line",
                                x0=0,
                                y0=conf_level,
                                x1=20,
                                y1=conf_level,
                                line=dict(color="red", dash="dash"),
                                row=i+1, col=j
                            )
                            
                            fig.add_shape(
                                type="line",
                                x0=0,
                                y0=-conf_level,
                                x1=20,
                                y1=-conf_level,
                                line=dict(color="red", dash="dash"),
                                row=i+1, col=j
                            )
                    
                    fig.update_layout(height=800, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Forecast insights
                st.markdown('<div class="sub-header">Forecast Insights</div>', unsafe_allow_html=True)
                
                # Calculate
                # Calculate forecast statistics
                final_value = forecast_data.iloc[-1]
                max_forecast = forecast_mean.max()
                min_forecast = forecast_mean.min()
                end_forecast = forecast_mean.iloc[-1]
                
                max_change = ((max_forecast - final_value) / final_value) * 100
                overall_change = ((end_forecast - final_value) / final_value) * 100
                
                # Forecast summary
                st.markdown(f"""
                <div class="highlight">
                    <strong>Forecast Summary:</strong><br>
                    Current storage: <span class="warning">{final_value:,.0f}</span> kilobarrels<br>
                    Forecasted storage in {forecast_horizon} months: <span class="warning">{end_forecast:,.0f}</span> kilobarrels ({overall_change:+.1f}%)<br>
                    Maximum forecasted storage: {max_forecast:,.0f} kilobarrels ({max_change:+.1f}%)<br>
                    Minimum forecasted storage: {min_forecast:,.0f} kilobarrels
                </div>
                """, unsafe_allow_html=True)
                
                # Forecast data table
                with st.expander("View Forecast Data Table"):
                    forecast_df = pd.DataFrame({
                        'Date': forecast_index,
                        'Forecast': forecast_mean.values,
                        'Lower CI': forecast_ci.iloc[:, 0].values,
                        'Upper CI': forecast_ci.iloc[:, 1].values
                    })
                    
                    st.dataframe(
                        forecast_df.set_index('Date').style.format({
                            'Forecast': '{:,.0f}',
                            'Lower CI': '{:,.0f}',
                            'Upper CI': '{:,.0f}'
                        })
                    )
                
                # Strategic implications
                risk_level = "High" if abs(max_change) > 20 else "Medium" if abs(max_change) > 10 else "Low"
                trend_direction = "increasing" if overall_change > 0 else "decreasing"
                
                # Replace with:
                recommendations = []
                recommendations.append(f"The forecast suggests an overall <span class=\"{'success' if trend_direction == 'increasing' else 'warning'}\">{trend_direction}</span> trend in storage levels for {selected_region}.")
                recommendations.append(f"Volatility risk: <span class=\"{risk_level.lower()}\">{risk_level}</span>")

                if max_change > 15:
                    recommendations.append("Storage may approach capacity constraints. Consider securing additional storage options.")
                if overall_change < -15:
                    recommendations.append("Storage may decrease significantly. Consider optimizing existing capacity.")

                recommendations.append(f"Recommendation: {'Prepare for expanding storage needs' if overall_change > 0 else 'Monitor decreasing storage trend'}")

                recommendations_html = "".join([f"<li>{item}</li>" for item in recommendations])

                st.markdown(f"""
                <div class="highlight">
                    <strong>Strategic Implications:</strong><br>
                    <ul>
                        {recommendations_html}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif model_type == "Exponential Smoothing":
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                
                # Fit model
                model = ExponentialSmoothing(
                    forecast_data,
                    trend=trend_type,
                    seasonal=seasonal_type,
                    seasonal_periods=seasonal_periods
                )
                model_fit = model.fit()
                
                # Generate forecast
                forecast_values = model_fit.forecast(forecast_horizon)
                
                # Create forecast index
                forecast_index = pd.date_range(start=forecast_data.index[-1], periods=forecast_horizon + 1, freq="MS")[1:]
                
                # Create prediction intervals manually
                resid_std = np.std(model_fit.resid)
                z_value = stats.norm.ppf(1 - (1 - confidence_level/100) / 2)
                
                forecast_ci_lower = forecast_values - z_value * resid_std
                forecast_ci_upper = forecast_values + z_value * resid_std
                
                # Create figure
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add fitted values
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=model_fit.fittedvalues,
                    mode='lines',
                    name='Fitted',
                    line=dict(color='green')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast_values.values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                    y=forecast_ci_lower.tolist() + forecast_ci_upper.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(231,107,243,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Storage Forecast for {selected_region} using Exponential Smoothing",
                    xaxis_title="Date",
                    yaxis_title="Storage (KiloBarrels)",
                    height=500
                )
                
                forecast_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Model performance
                st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
                
                # Calculate error metrics
                mse = ((model_fit.fittedvalues - forecast_data) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(model_fit.fittedvalues - forecast_data).mean()
                mape = np.abs((model_fit.fittedvalues - forecast_data) / forecast_data).mean() * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Error Metrics:</strong><br>
                        RMSE: {rmse:.2f}<br>
                        MAE: {mae:.2f}<br>
                        MAPE: {mape:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Model parameters
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Model Configuration:</strong><br>
                        Trend: {trend_type if trend_type else 'None'}<br>
                        Seasonality: {seasonal_type if seasonal_type else 'None'}<br>
                        Seasonal Periods: {seasonal_periods if seasonal_type else 'N/A'}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Forecast insights
                st.markdown('<div class="sub-header">Forecast Insights</div>', unsafe_allow_html=True)
                
                # Calculate forecast statistics
                final_value = forecast_data.iloc[-1]
                max_forecast = forecast_values.max()
                min_forecast = forecast_values.min()
                end_forecast = forecast_values.iloc[-1]
                
                max_change = ((max_forecast - final_value) / final_value) * 100
                overall_change = ((end_forecast - final_value) / final_value) * 100
                
                # Forecast summary
                st.markdown(f"""
                <div class="highlight">
                    <strong>Forecast Summary:</strong><br>
                    Current storage: <span class="warning">{final_value:,.0f}</span> kilobarrels<br>
                    Forecasted storage in {forecast_horizon} months: <span class="warning">{end_forecast:,.0f}</span> kilobarrels ({overall_change:+.1f}%)<br>
                    Maximum forecasted storage: {max_forecast:,.0f} kilobarrels ({max_change:+.1f}%)<br>
                    Minimum forecasted storage: {min_forecast:,.0f} kilobarrels
                </div>
                """, unsafe_allow_html=True)
                
                # Forecast data table
                with st.expander("View Forecast Data Table"):
                    forecast_df = pd.DataFrame({
                        'Date': forecast_index,
                        'Forecast': forecast_values.values,
                        'Lower CI': forecast_ci_lower,
                        'Upper CI': forecast_ci_upper
                    })
                    
                    st.dataframe(
                        forecast_df.set_index('Date').style.format({
                            'Forecast': '{:,.0f}',
                            'Lower CI': '{:,.0f}',
                            'Upper CI': '{:,.0f}'
                        })
                    )
                
                # Seasonal patterns
                if seasonal_type:
                    try:
                        seasonal_components = model_fit.seasonal_
                        if len(seasonal_components) > 0:
                            fig = px.line(
                                x=range(1, len(seasonal_components) + 1),
                                y=seasonal_components,
                                title="Seasonal Components",
                                markers=True
                            )
                            
                            fig.update_layout(
                                xaxis_title="Season",
                                yaxis_title="Effect",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    except (AttributeError, TypeError):
                        st.info("Seasonal components are not available for this model configuration.")
            
            elif model_type == "Moving Average":
                # Calculate moving average
                ma_values = forecast_data.rolling(window=ma_window).mean()
                
                # For forecasting, use the last MA value
                last_ma = ma_values.iloc[-1]
                forecast_values = pd.Series([last_ma] * forecast_horizon)
                
                # Create forecast index
                forecast_index = pd.date_range(start=forecast_data.index[-1], periods=forecast_horizon + 1, freq="MS")[1:]
                
                # Calculate simple prediction intervals
                ma_std = forecast_data.rolling(window=ma_window).std().iloc[-1]
                z_value = stats.norm.ppf(1 - (1 - confidence_level/100) / 2)
                
                forecast_ci_lower = [last_ma - z_value * ma_std] * forecast_horizon
                forecast_ci_upper = [last_ma + z_value * ma_std] * forecast_horizon
                
                # Create figure
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=forecast_data.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add moving average
                fig.add_trace(go.Scatter(
                    x=forecast_data.index,
                    y=ma_values.values,
                    mode='lines',
                    name=f'{ma_window}-Month MA',
                    line=dict(color='green')
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_index,
                    y=forecast_values.values,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=forecast_index.tolist() + forecast_index.tolist()[::-1],
                    y=forecast_ci_lower + forecast_ci_upper[::-1],
                    fill='toself',
                    fillcolor='rgba(231,107,243,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{confidence_level}% Confidence Interval'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"Storage Forecast for {selected_region} using {ma_window}-Month Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Storage (KiloBarrels)",
                    height=500
                )
                
                forecast_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Model performance
                st.markdown('<div class="sub-header">Moving Average Performance</div>', unsafe_allow_html=True)
                
                # Calculate error metrics where MA values are available
                valid_indices = ~np.isnan(ma_values)
                mse = ((ma_values[valid_indices] - forecast_data[valid_indices]) ** 2).mean()
                rmse = np.sqrt(mse)
                mae = np.abs(ma_values[valid_indices] - forecast_data[valid_indices]).mean()
                mape = np.abs((ma_values[valid_indices] - forecast_data[valid_indices]) / forecast_data[valid_indices]).mean() * 100
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Error Metrics:</strong><br>
                        RMSE: {rmse:.2f}<br>
                        MAE: {mae:.2f}<br>
                        MAPE: {mape:.2f}%
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # MA parameters
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Moving Average Configuration:</strong><br>
                        Window Size: {ma_window} months<br>
                        Last MA Value: {last_ma:,.0f} kilobarrels<br>
                        Standard Deviation: {ma_std:,.0f} kilobarrels
                    </div>
                    """, unsafe_allow_html=True)
                
                # Forecast insights
                st.markdown('<div class="sub-header">Forecast Insights</div>', unsafe_allow_html=True)
                
                # Calculate forecast statistics
                final_value = forecast_data.iloc[-1]
                max_forecast = last_ma
                min_forecast = last_ma
                end_forecast = last_ma
                
                overall_change = ((end_forecast - final_value) / final_value) * 100
                
                # Forecast summary
                st.markdown(f"""
                <div class="highlight">
                    <strong>Forecast Summary:</strong><br>
                    Current storage: <span class="warning">{final_value:,.0f}</span> kilobarrels<br>
                    Moving average ({ma_window}-month): <span class="warning">{last_ma:,.0f}</span> kilobarrels<br>
                    Current vs. MA: {overall_change:+.1f}%<br><br>
                    
                    <strong>Note:</strong> The moving average forecast projects flat storage at {last_ma:,.0f} kilobarrels for the next {forecast_horizon} months. 
                    This simple model doesn't account for trends or seasonal patterns.
                </div>
                """, unsafe_allow_html=True)
                
                # Forecast data table
                with st.expander("View Forecast Data Table"):
                    forecast_df = pd.DataFrame({
                        'Date': forecast_index,
                        'Forecast': forecast_values.values,
                        'Lower CI': forecast_ci_lower,
                        'Upper CI': forecast_ci_upper
                    })
                    
                    st.dataframe(
                        forecast_df.set_index('Date').style.format({
                            'Forecast': '{:,.0f}',
                            'Lower CI': '{:,.0f}',
                            'Upper CI': '{:,.0f}'
                        })
                    )
            
            # Model comparison
            if st.checkbox("Compare with other models"):
                st.markdown('<div class="sub-header">Model Comparison</div>', unsafe_allow_html=True)
                
                with st.spinner("Comparing models..."):
                    # Simple function to fit and evaluate models
                    def evaluate_model(model_name):
                        if model_name == "SARIMA":
                            # Simple SARIMA
                            model = SARIMAX(forecast_data, order=(1,1,1), seasonal_order=(1,1,1,12))
                            model_fit = model.fit(disp=False)
                            predictions = model_fit.fittedvalues
                        elif model_name == "Exponential Smoothing":
                            # Simple Exp Smoothing
                            model = ExponentialSmoothing(forecast_data, trend='add', seasonal='add', seasonal_periods=12)
                            model_fit = model.fit()
                            predictions = model_fit.fittedvalues
                        elif model_name == "Moving Average":
                            # 3-month MA
                            predictions = forecast_data.rolling(window=3).mean()
                        
                        # Calculate metrics
                        valid_indices = ~np.isnan(predictions)
                        
                        if valid_indices.sum() > 0:
                            mse = ((predictions[valid_indices] - forecast_data[valid_indices]) ** 2).mean()
                            rmse = np.sqrt(mse)
                            mae = np.abs(predictions[valid_indices] - forecast_data[valid_indices]).mean()
                            mape = np.abs((predictions[valid_indices] - forecast_data[valid_indices]) / forecast_data[valid_indices]).mean() * 100
                            
                            return {
                                "Model": model_name,
                                "RMSE": rmse,
                                "MAE": mae,
                                "MAPE": mape
                            }
                        else:
                            return {
                                "Model": model_name,
                                "RMSE": np.nan,
                                "MAE": np.nan,
                                "MAPE": np.nan
                            }
                    
                    # Evaluate all models
                    models_to_compare = ["SARIMA", "Exponential Smoothing", "Moving Average"]
                    comparison_results = []
                    
                    for model_name in models_to_compare:
                        result = evaluate_model(model_name)
                        comparison_results.append(result)
                    
                    # Create comparison table
                    comparison_df = pd.DataFrame(comparison_results)
                    
                    # Highlight best model
                    def highlight_min(s):
                        is_min = s == s.min()
                        return ['background-color: #90EE90' if v else '' for v in is_min]
                    
                    st.dataframe(
                        comparison_df.set_index("Model").style.format({
                            "RMSE": "{:.2f}",
                            "MAE": "{:.2f}",
                            "MAPE": "{:.2f}%"
                        })
                        .apply(highlight_min)
                    )
                    
                    # Model recommendation
                    best_model = comparison_df.iloc[comparison_df["RMSE"].argmin()]["Model"]
                    
                    st.markdown(f"""
                    <div class="highlight">
                        <strong>Model Recommendation:</strong><br>
                        Based on error metrics, the <span class="success">{best_model}</span> model appears to perform best for this data.
                        {'<br>This aligns with your current model selection.' if best_model == model_type else '<br>Consider switching to this model for potentially better forecasts.'}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download forecast data
            csv = forecast_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download Forecast Data",
                data=csv,
                file_name=f"{selected_region}_storage_forecast.csv",
                mime="text/csv",
            )
        
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.text("Try adjusting your model parameters or selecting a different region with more data.")

# Anomaly Detection Page
elif page == "Anomaly Detection":
    st.markdown('<div class="main-header">Storage Anomaly Detection</div>', unsafe_allow_html=True)
    
    # Anomaly detection method selector
    detection_method = st.selectbox(
        "Select Detection Method", 
        ["Z-Score", "IQR Method", "Moving Average Deviation", "Seasonal Decomposition"]
    )
    
    # Common parameters
    col1, col2 = st.columns(2)
    
    with col1:
        # Sensitivity parameter
        if detection_method == "Z-Score":
            threshold = st.slider("Z-Score Threshold", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        elif detection_method == "IQR Method":
            threshold = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        elif detection_method == "Moving Average Deviation":
            threshold = st.slider("Deviation Threshold (σ)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
        elif detection_method == "Seasonal Decomposition":
            threshold = st.slider("Residual Threshold (σ)", min_value=1.5, max_value=5.0, value=3.0, step=0.1)
    
    with col2:
        # Time window parameter
        if detection_method in ["Z-Score", "IQR Method"]:
            lookback = st.slider("Lookback Period (Months)", min_value=6, max_value=36, value=12)
        elif detection_method == "Moving Average Deviation":
            window_size = st.slider("Moving Average Window", min_value=3, max_value=12, value=3)
        elif detection_method == "Seasonal Decomposition":
            seasonal_period = st.slider("Seasonal Period (Months)", min_value=3, max_value=12, value=12)
    
    # Analysis target selector
    analysis_target = st.radio(
        "Analyze",
        ["Storage Levels", "Storage Changes (MoM)", "Floating vs Terminal Storage Ratio"],
        horizontal=True
    )
    
    # Filter for detected anomalies checkbox
    show_only_anomalies = st.checkbox("Show Only Detected Anomalies", value=True)
    
    # Process data and detect anomalies
    st.markdown('<div class="sub-header">Detected Anomalies</div>', unsafe_allow_html=True)
    
    # Prepare data based on selected target
    if analysis_target == "Storage Levels":
        # Use total storage values
        analysis_data = filtered_data.copy()
        value_column = "TotalStorage_KiloBarrels"
        y_axis_label = "Storage (KiloBarrels)"
        
    elif analysis_target == "Storage Changes (MoM)":
        # Calculate month-over-month changes
        analysis_data = filtered_data.copy()
        analysis_data["MoM_Change"] = analysis_data.groupby("CountryName")["TotalStorage_KiloBarrels"].pct_change() * 100
        value_column = "MoM_Change"
        y_axis_label = "Monthly Change (%)"
        
    elif analysis_target == "Floating vs Terminal Storage Ratio":
        # Calculate ratio of floating to terminal storage
        analysis_data = filtered_data.copy()
        # Avoid division by zero
        analysis_data["Storage_Ratio"] = analysis_data.apply(
            lambda row: row["FloatingStorage_KiloBarrels"] / row["ObservedValue"] 
            if row["ObservedValue"] > 0 else 0, axis=1
        )
        value_column = "Storage_Ratio"
        y_axis_label = "Floating/Terminal Ratio"
    
    # Filter out missing values
    analysis_data = analysis_data.dropna(subset=[value_column])
    
    # Function to detect anomalies based on selected method
    def detect_anomalies(data, country, method, value_col):
        country_data = data[data["CountryName"] == country].copy()
        
        if len(country_data) < 3:  # Minimum data points needed
            return pd.DataFrame()
        
        country_data = country_data.sort_values("ReferenceDate")
        
        # Apply detection method
        if method == "Z-Score":
            # Calculate rolling mean and std
            if len(country_data) >= lookback:
                country_data["Rolling_Mean"] = country_data[value_col].rolling(window=lookback, min_periods=3).mean()
                country_data["Rolling_Std"] = country_data[value_col].rolling(window=lookback, min_periods=3).std()
                
                # Calculate z-scores
                country_data["Anomaly_Score"] = (country_data[value_col] - country_data["Rolling_Mean"]) / country_data["Rolling_Std"]
                
                # Flag anomalies
                country_data["Is_Anomaly"] = abs(country_data["Anomaly_Score"]) > threshold
            else:
                # If not enough data for rolling window, use global stats
                mean_val = country_data[value_col].mean()
                std_val = country_data[value_col].std()
                
                # Add this check:
                if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
                    return pd.DataFrame()
                    
                country_data["Anomaly_Score"] = (country_data[value_col] - mean_val) / std_val
                country_data["Is_Anomaly"] = abs(country_data["Anomaly_Score"]) > threshold
        
        elif method == "IQR Method":
            # Calculate rolling IQR
            if len(country_data) >= lookback:
                def rolling_iqr(values):
                    q1 = np.nanpercentile(values, 25)
                    q3 = np.nanpercentile(values, 75)
                    return q3 - q1
                
                country_data["Rolling_Median"] = country_data[value_col].rolling(window=lookback, min_periods=3).median()
                country_data["Rolling_IQR"] = country_data[value_col].rolling(window=lookback, min_periods=3).apply(rolling_iqr)
                
                # Calculate upper and lower bounds
                country_data["Upper_Bound"] = country_data["Rolling_Median"] + threshold * country_data["Rolling_IQR"]
                country_data["Lower_Bound"] = country_data["Rolling_Median"] - threshold * country_data["Rolling_IQR"]
                
                # Flag anomalies
                country_data["Is_Anomaly"] = (country_data[value_col] > country_data["Upper_Bound"]) | (country_data[value_col] < country_data["Lower_Bound"])
                
                # Calculate normalized score for consistency
                country_data["Anomaly_Score"] = (country_data[value_col] - country_data["Rolling_Median"]) / (country_data["Rolling_IQR"] + 1e-10)
            else:
                # Global IQR if not enough data
                q1 = country_data[value_col].quantile(0.25)
                q3 = country_data[value_col].quantile(0.75)
                iqr = q3 - q1
                
                upper_bound = q3 + threshold * iqr
                lower_bound = q1 - threshold * iqr
                
                country_data["Anomaly_Score"] = (country_data[value_col] - country_data[value_col].median()) / (iqr + 1e-10)
                country_data["Is_Anomaly"] = (country_data[value_col] > upper_bound) | (country_data[value_col] < lower_bound)
        
        elif method == "Moving Average Deviation":
            # Calculate moving average
            country_data["MA"] = country_data[value_col].rolling(window=window_size, min_periods=2).mean()
            
            # Calculate deviation from MA
            country_data["Deviation"] = country_data[value_col] - country_data["MA"]
            
            # Calculate rolling standard deviation of deviations
            country_data["Deviation_Std"] = country_data["Deviation"].rolling(window=max(window_size*3, 6), min_periods=3).std()
            
            # Flag anomalies
            country_data["Anomaly_Score"] = country_data["Deviation"] / (country_data["Deviation_Std"] + 1e-10)
            country_data["Is_Anomaly"] = abs(country_data["Anomaly_Score"]) > threshold
        
        elif method == "Seasonal Decomposition":
            # Need enough data for seasonal decomposition
            if len(country_data) >= 2 * seasonal_period:
                # Set date as index
                ts_data = country_data.set_index("ReferenceDate")[value_col]
                
                # Ensure regular frequency
                ts_data = ts_data.asfreq('MS', method='ffill')
                
                try:
                    # Perform decomposition
                    decomposition = seasonal_decompose(ts_data, model='additive', period=seasonal_period)
                    
                    # Get residuals
                    residuals = decomposition.resid
                    
                    # Calculate standard deviation of residuals
                    resid_std = residuals.std()
                    
                    # Add back to data
                    country_data = country_data.copy()
                    country_data["Residual"] = [residuals.get(date, np.nan) for date in country_data["ReferenceDate"]]
                    
                    # Flag anomalies
                    country_data["Anomaly_Score"] = country_data["Residual"] / (resid_std + 1e-10)
                    country_data["Is_Anomaly"] = abs(country_data["Anomaly_Score"]) > threshold
                except:
                    # If decomposition fails, return empty DataFrame
                    return pd.DataFrame()
            else:
                # Fall back to a simpler method when not enough data
                mean_val = country_data[value_col].mean()
                std_val = country_data[value_col].std()
                
                if std_val > 0:  # Avoid division by zero
                    country_data["Anomaly_Score"] = (country_data[value_col] - mean_val) / std_val
                    country_data["Is_Anomaly"] = abs(country_data["Anomaly_Score"]) > threshold
                    return country_data
                else:
                    return pd.DataFrame()
        
        # Return anomalies or all data
        if show_only_anomalies:
            return country_data[country_data["Is_Anomaly"]]
        else:
            return country_data
    
    # Process each region
    all_anomalies = []
    
    for region in selected_regions:
        region_anomalies = detect_anomalies(analysis_data, region, detection_method, value_column)
        if not region_anomalies.empty:
            all_anomalies.append(region_anomalies)
    
    # Combine results
    if all_anomalies:
        combined_anomalies = pd.concat(all_anomalies)
        combined_anomalies = combined_anomalies.sort_values(["ReferenceDate", "CountryName"], ascending=[False, True])
        
        # Display results table
        if not combined_anomalies.empty:
            # Select relevant columns
            display_columns = ["CountryName", "ReferenceDate", value_column, "Anomaly_Score"]
            display_df = combined_anomalies[display_columns].copy()
            
            # Rename columns for display
            column_mapping = {
                "CountryName": "Region",
                "ReferenceDate": "Date",
                "TotalStorage_KiloBarrels": "Storage (KB)",
                "MoM_Change": "Monthly Change (%)",
                "Storage_Ratio": "Floating/Terminal Ratio",
                "Anomaly_Score": "Anomaly Score"
            }
            
            # Apply relevant renames
            rename_dict = {k: v for k, v in column_mapping.items() if k in display_df.columns}
            display_df = display_df.rename(columns=rename_dict)
            
            # Custom format based on value type
            if value_column == "TotalStorage_KiloBarrels":
                format_dict = {"Storage (KB)": "{:,.0f}", "Anomaly Score": "{:+.2f}"}
            elif value_column == "MoM_Change":
                format_dict = {"Monthly Change (%)": "{:+.2f}%", "Anomaly Score": "{:+.2f}"}
            else:
                format_dict = {"Floating/Terminal Ratio": "{:.3f}", "Anomaly Score": "{:+.2f}"}
            
            display_df = display_df.reset_index(drop=True)
            styled_df = display_df.style.format(format_dict).background_gradient(cmap="RdYlGn_r", subset=["Anomaly Score"])
            st.dataframe(styled_df)
            
            # Count anomalies by region
            region_counts = combined_anomalies.groupby("CountryName").size().reset_index()
            region_counts.columns = ["Region", "Anomaly Count"]
            
            # Create bar chart of anomaly counts
            fig = px.bar(
                region_counts,
                x="Region",
                y="Anomaly Count",
                title="Anomalies by Region",
                color="Anomaly Count",
                color_continuous_scale="Reds"
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title="Number of Anomalies",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No anomalies detected with the current settings. Try adjusting the threshold or analysis parameters.")
    else:
        st.info(f"No anomalies detected with the current settings. Try adjusting the threshold or analysis parameters.")
    
    # Detailed analysis for a selected region
    st.markdown('<div class="sub-header">Region-Specific Anomaly Analysis</div>', unsafe_allow_html=True)
    
    selected_region = st.selectbox("Select Region for Detailed Analysis", selected_regions)
    
    # Get data for selected region
    region_data = analysis_data[analysis_data["CountryName"] == selected_region].copy()
    
    if not region_data.empty:
        # Apply detection method to get all data points
        region_analysis = detect_anomalies(analysis_data, selected_region, detection_method, value_column)
        
        if not region_analysis.empty:
            # Create visualization
            fig = go.Figure()
            
            # Add main data line
            fig.add_trace(go.Scatter(
                x=region_analysis["ReferenceDate"],
                y=region_analysis[value_column],
                mode="lines+markers",
                name=y_axis_label,
                line=dict(color="blue")
            ))
            
            # Add detected anomalies
            anomalies = region_analysis[region_analysis["Is_Anomaly"]]
            
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies["ReferenceDate"],
                    y=anomalies[value_column],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(
                        size=12,
                        color="red",
                        symbol="circle-open",
                        line=dict(width=2)
                    )
                ))
            
            # Add reference lines based on detection method
            if detection_method == "Z-Score" and "Rolling_Mean" in region_analysis.columns:
                # Add mean line
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=region_analysis["Rolling_Mean"],
                    mode="lines",
                    name="Rolling Mean",
                    line=dict(color="green", dash="dash")
                ))
                
                # Add threshold lines
                upper_threshold = region_analysis["Rolling_Mean"] + threshold * region_analysis["Rolling_Std"]
                lower_threshold = region_analysis["Rolling_Mean"] - threshold * region_analysis["Rolling_Std"]
                
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=upper_threshold,
                    mode="lines",
                    name=f"+{threshold}σ Threshold",
                    line=dict(color="red", dash="dot")
                ))
                
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=lower_threshold,
                    mode="lines",
                    name=f"-{threshold}σ Threshold",
                    line=dict(color="red", dash="dot")
                ))
            
            elif detection_method == "IQR Method" and "Rolling_Median" in region_analysis.columns:
                # Add median line
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=region_analysis["Rolling_Median"],
                    mode="lines",
                    name="Rolling Median",
                    line=dict(color="green", dash="dash")
                ))
                
                # Add threshold lines
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=region_analysis["Upper_Bound"],
                    mode="lines",
                    name="Upper Threshold",
                    line=dict(color="red", dash="dot")
                ))
                
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=region_analysis["Lower_Bound"],
                    mode="lines",
                    name="Lower Threshold",
                    line=dict(color="red", dash="dot")
                ))
            
            elif detection_method == "Moving Average Deviation" and "MA" in region_analysis.columns:
                # Add MA line
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=region_analysis["MA"],
                    mode="lines",
                    name=f"{window_size}-Period MA",
                    line=dict(color="green", dash="dash")
                ))
                
                # Add threshold lines
                upper_threshold = region_analysis["MA"] + threshold * region_analysis["Deviation_Std"]
                lower_threshold = region_analysis["MA"] - threshold * region_analysis["Deviation_Std"]
                
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=upper_threshold,
                    mode="lines",
                    name=f"+{threshold}σ Threshold",
                    line=dict(color="red", dash="dot")
                ))
                
                fig.add_trace(go.Scatter(
                    x=region_analysis["ReferenceDate"],
                    y=lower_threshold,
                    mode="lines",
                    name=f"-{threshold}σ Threshold",
                    line=dict(color="red", dash="dot")
                ))
            
            # Update layout
            fig.update_layout(
                title=f"{analysis_target} for {selected_region} with Detected Anomalies",
                xaxis_title="Date",
                yaxis_title=y_axis_label,
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly score chart
            if "Anomaly_Score" in region_analysis.columns:
                fig = px.line(
                    region_analysis,
                    x="ReferenceDate",
                    y="Anomaly_Score",
                    title=f"Anomaly Scores for {selected_region}",
                    markers=True
                )
                
                # Add threshold lines
                fig.add_shape(
                    type="line",
                    x0=region_analysis["ReferenceDate"].min(),
                    y0=threshold,
                    x1=region_analysis["ReferenceDate"].max(),
                    y1=threshold,
                    line=dict(color="red", dash="dash")
                )
                
                fig.add_shape(
                    type="line",
                    x0=region_analysis["ReferenceDate"].min(),
                    y0=-threshold,
                    x1=region_analysis["ReferenceDate"].max(),
                    y1=-threshold,
                    line=dict(color="red", dash="dash")
                )
                
                # Highlight anomalies
                anomalies = region_analysis[region_analysis["Is_Anomaly"]]
                
                if not anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies["ReferenceDate"],
                        y=anomalies["Anomaly_Score"],
                        mode="markers",
                        name="Anomalies",
                        marker=dict(
                            size=12,
                            color="red",
                            symbol="circle-open",
                            line=dict(width=2)
                        )
                    ))
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Anomaly Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed analysis of each anomaly
            if "Is_Anomaly" in region_analysis.columns:
                anomalies = region_analysis[region_analysis["Is_Anomaly"]]
                
                if not anomalies.empty:
                    st.markdown('<div class="sub-header">Detailed Anomaly Analysis</div>', unsafe_allow_html=True)
                    
                    for i, (_, anomaly) in enumerate(anomalies.iterrows()):
                        date = anomaly["ReferenceDate"].strftime("%B %Y")
                        value = anomaly[value_column]
                        score = anomaly["Anomaly_Score"]
                        
                        # Determine if high or low anomaly
                        direction = "high" if score > 0 else "low"
                        
                        # Format value based on analysis target
                        if value_column == "TotalStorage_KiloBarrels":
                            value_formatted = f"{value:,.0f} kilobarrels"
                        elif value_column == "MoM_Change":
                            value_formatted = f"{value:+.2f}%"
                        else:
                            value_formatted = f"{value:.3f}"
                        
                        # Build explanation based on context
                        if value_column == "TotalStorage_KiloBarrels":
                            explanation = f"""
                            <div class="highlight">
                                <strong>Anomaly {i+1}: {date}</strong><br>
                                <span class="{'warning' if direction == 'high' else 'success'}">{direction.title()} storage anomaly</span> detected.<br>
                                Storage level: <span class="warning">{value_formatted}</span><br>
                                Anomaly score: {score:.2f} ({abs(score):.1f}x threshold)<br>
                                <br>
                                <strong>Possible Causes:</strong>
                                <ul>
                                    {"<li>Significant inventory build due to supply glut or lower demand</li>" if direction == "high" else ""}
                                    {"<li>Possible data reporting error or change in measurement methodology</li>" if abs(score) > 4 else ""}
                                    {"<li>New storage capacity coming online</li>" if direction == "high" else ""}
                                    {"<li>Significant drawdown due to high demand or supply constraint</li>" if direction == "low" else ""}
                                    {"<li>Possible operational issues or maintenance</li>" if direction == "low" else ""}
                                </ul>
                            </div>
                            """
                        elif value_column == "MoM_Change":
                            explanation = f"""
                            <div class="highlight">
                                <strong>Anomaly {i+1}: {date}</strong><br>
                                <span class="{'warning' if direction == 'high' else 'success'}">{direction.title()} change rate anomaly</span> detected.<br>
                                Monthly change: <span class="warning">{value_formatted}</span><br>
                                Anomaly score: {score:.2f} ({abs(score):.1f}x threshold)<br>
                                <br>
                                <strong>Possible Causes:</strong>
                                <ul>
                                    {"<li>Rapid inventory build possibly due to market contango</li>" if direction == "high" else ""}
                                    {"<li>Sudden shift in regional trade flows</li>" if abs(score) > 3 else ""}
                                    {"<li>Policy change or regulatory impact</li>" if abs(score) > 3 else ""}
                                    {"<li>Rapid inventory drawdown possibly due to tight supply</li>" if direction == "low" else ""}
                                    {"<li>Seasonal factors outside normal patterns</li>" if abs(score) > 2 else ""}
                                </ul>
                            </div>
                            """
                        else:  # Storage ratio
                            explanation = f"""
                            <div class="highlight">
                                <strong>Anomaly {i+1}: {date}</strong><br>
                                <span class="{'warning' if direction == 'high' else 'success'}">{direction.title()} floating-to-terminal ratio anomaly</span> detected.<br>
                                Ratio: <span class="warning">{value_formatted}</span><br>
                                Anomaly score: {score:.2f} ({abs(score):.1f}x threshold)<br>
                                <br>
                                <strong>Possible Causes:</strong>
                                <ul>
                                    {"<li>Terminal storage constraints pushing inventory to floating storage</li>" if direction == "high" else ""}
                                    {"<li>Contango market structure incentivizing floating storage</li>" if direction == "high" else ""}
                                    {"<li>Shift from floating to terminal storage due to backwardation</li>" if direction == "low" else ""}
                                    {"<li>Increased terminal capacity or throughput</li>" if direction == "low" else ""}
                                    {"<li>Change in regional trade patterns</li>" if abs(score) > 2 else ""}
                                </ul>
                            </div>
                            """
                        
                        st.markdown(explanation, unsafe_allow_html=True)
                else:
                    st.info(f"No anomalies detected for {selected_region} with the current settings.")
            else:
                st.info(f"Insufficient data for anomaly detection in {selected_region}.")
        else:
            st.info(f"Insufficient data for anomaly detection in {selected_region}.")
    else:
        st.warning(f"No data available for {selected_region}.")
    
    # Advanced settings
    with st.expander("Advanced Settings & Documentation"):
        st.markdown("""
        ### Anomaly Detection Methods
        
        #### Z-Score Method
        Detects values that deviate significantly from the mean in terms of standard deviations. A rolling window approach is used to account for evolving data patterns.
        - **Threshold**: Number of standard deviations from the mean to flag as anomaly
        - **Lookback Period**: Number of months to include in the rolling window
        
        #### IQR (Interquartile Range) Method
        Uses the interquartile range to identify outliers, which is more robust to extreme values than Z-score.
        - **IQR Multiplier**: Factor to multiply the IQR by to determine thresholds
        - **Lookback Period**: Number of months to include in the rolling window
        
        #### Moving Average Deviation
        Identifies values that deviate significantly from a moving average trend.
        - **Deviation Threshold**: Number of standard deviations from the MA to flag as anomaly
        - **Moving Average Window**: Number of periods to include in the moving average calculation
        
        #### Seasonal Decomposition
        Decomposes time series into trend, seasonal, and residual components, then identifies anomalies in the residuals.
        - **Residual Threshold**: Number of standard deviations in the residual component to flag as anomaly
        - **Seasonal Period**: Number of months in the seasonal cycle
        
        ### Analysis Targets
        
        - **Storage Levels**: Detects anomalies in absolute storage volumes
        - **Storage Changes**: Detects anomalies in month-over-month percentage changes
        - **Floating vs Terminal Ratio**: Detects anomalies in the ratio between floating and terminal storage
        
        ### Using Anomaly Detection for Trading Insights
        
        Anomalies can provide valuable trading signals:
        - **Supply/Demand Imbalances**: Unusual storage builds or draws may indicate changing market dynamics
        - **Market Structure Shifts**: Changes in floating-to-terminal ratios can signal shifts between contango and backwardation
        - **Regional Arbitrage Opportunities**: Anomalies in specific regions may highlight arbitrage opportunities
        - **Early Warning Signals**: Anomalies often precede larger market movements
        
        ### Validation
        
        Always validate detected anomalies against known market events and fundamental factors before making trading decisions.
        """)
