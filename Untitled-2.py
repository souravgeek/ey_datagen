import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile

# Set page config
st.set_page_config(
    page_title="Telecom Synthetic Data Generator",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_telecom_synthetic_data(
    # Geographic parameters
    num_states,
    districts_per_state,
    blocks_per_district,
    gps_per_block,
    villages_per_gp,
    connections_per_village,
    
    # Tuning parameters for variability
    infra_quality_variance,
    sla_baseline_mean,
    sla_variance,
    causal_strength,
    outlier_probability,
    days_active_causation,
    admin_nw_different_pct,  # % of GPs where admin and nw blocks differ
    
    # Temporal parameters
    sla_month,
    
    # FTTH parameters
    bandwidth_options,
    disconnection_rate,
    
    # Progress callback
    progress_callback=None
):
    """
    Generate synthetic telecom data with configurable causal relationships
    """
    
    np.random.seed(42)  # For reproducible results
    random.seed(42)
    
    if progress_callback:
        progress_callback(0.1, "Initializing data structures...")
    
    # Indian state names for realism
    state_names = [
        "Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", 
        "Gujarat", "Rajasthan", "West Bengal", "Andhra Pradesh",
        "Madhya Pradesh", "Odisha"
    ][:num_states]
    
    # Helper function to generate unique Indian place names
    def generate_place_name(prefix, index):
        suffixes = ["pur", "ganj", "abad", "nagar", "puram", "garh", "kota", "gram"]
        prefixes_list = ["Raj", "Sunder", "Chandra", "Krishna", "Rama", "Vijay", "Anil", "Mukesh", 
                        "Gopal", "Hari", "Shyam", "Ravi", "Kiran", "Deepak", "Suresh", "Ramesh",
                        "Mahesh", "Ganesh", "Dinesh", "Naresh", "Yogesh", "Umesh", "Kamal", "Vimal"]
        return f"{random.choice(prefixes_list)}{prefix}{index:04d}{random.choice(suffixes)}"
    
    # ===========================================
    # 1. GENERATE INFRASTRUCTURE DATA (infra_df)
    # ===========================================
    
    if progress_callback:
        progress_callback(0.2, "Generating infrastructure data...")
    
    infra_data = []
    location_hierarchy = {}  # Store for later use
    
    # Global counters for unique codes
    state_code_counter = 1
    district_code_counter = 1
    block_code_counter = 1
    
    for state_idx, state_name in enumerate(state_names):
        state_code = state_code_counter
        state_code_counter += 1
        
        # Add some variance to districts per state
        num_districts = max(3, int(np.random.normal(districts_per_state, districts_per_state*0.2)))
        
        for dist_idx in range(num_districts):
            district_code = district_code_counter
            district_code_counter += 1
            district_name = generate_place_name("Dist", district_code)
            
            # Add variance to blocks per district
            num_blocks = max(5, int(np.random.normal(blocks_per_district, blocks_per_district*0.2)))
            
            for block_idx in range(num_blocks):
                block_code = block_code_counter
                block_code_counter += 1
                block_name = generate_place_name("Block", block_code)
                
                # Generate infrastructure quality with regional clustering
                # Some states/districts might have systematically better/worse infrastructure
                regional_bias = np.random.normal(0, 0.1)  # Regional infrastructure bias
                base_quality = 0.7 + regional_bias  # Base infrastructure quality
                
                # Apply variance
                quality_modifier = np.random.normal(0, infra_quality_variance)
                actual_quality = np.clip(base_quality + quality_modifier, 0.1, 0.95)
                
                # Generate infrastructure components based on quality
                power_connection = np.random.random() < (0.85 + actual_quality*0.1)
                power_working = power_connection and (np.random.random() < (0.7 + actual_quality*0.25))
                
                ac_presence = np.random.random() < (0.6 + actual_quality*0.3)
                ac_working = ac_presence and (np.random.random() < (0.65 + actual_quality*0.3))
                
                generator_presence = np.random.random() < (0.4 + actual_quality*0.4)
                generator_working = generator_presence and (np.random.random() < (0.75 + actual_quality*0.2))
                
                manpower_presence = np.random.random() < (0.8 + actual_quality*0.15)
                
                # Battery capacity influenced by infrastructure quality
                battery_capacity = np.random.uniform(4, 8) * (0.7 + actual_quality*0.6)
                battery_capacity = round(battery_capacity, 1)
                
                # Add outliers
                if np.random.random() < outlier_probability:
                    battery_capacity = np.random.choice([1, 2, 12, 15])  # Extreme values
                
                infra_record = {
                    'STATE_CODE': state_code,
                    'DISTIRCT_CODE': district_code,
                    'BLOCK_CODE': block_code,
                    'STATE': state_name,
                    'DISTRICT': district_name,
                    'BLOCK': block_name,
                    'POWER_CONNECTION': power_connection,
                    'POWER_WORKING': power_working,
                    'AC_PRESENCE': ac_presence,
                    'AC_WORKING': ac_working,
                    'GENERATOR_PRESENCE': generator_presence,
                    'GENERATOR_WORKING': generator_working,
                    'MANPOWER_PRESENCE': manpower_presence,
                    'BATTERY_CAPACITY': battery_capacity,
                    '_quality_score': actual_quality  # Internal use for causality
                }
                
                infra_data.append(infra_record)
                
                # Store hierarchy for SLA generation
                location_hierarchy[(state_code, district_code, block_code)] = {
                    'state_name': state_name,
                    'district_name': district_name,
                    'block_name': block_name,
                    'quality_score': actual_quality
                }
    
    infra_df = pd.DataFrame(infra_data)
    
    # ===========================================
    # 2. GENERATE SLA DATA (sla_df)
    # ===========================================
    
    if progress_callback:
        progress_callback(0.4, "Generating SLA performance data...")
    
    sla_data = []
    gp_hierarchy = {}  # Store GP info for FTTH generation
    
    # Global GP counter for unique codes
    gp_code_counter = 1
    
    # Generate date columns for May 2025 with actual dates
    start_date = datetime.strptime(sla_month, "%Y-%m")
    days_in_month = 31 if sla_month.endswith("-05") else 30
    date_columns = []
    for day in range(1, days_in_month + 1):
        date_obj = start_date.replace(day=day)
        date_columns.append(date_obj.strftime("%d %b %Y"))
    
    for _, infra_row in infra_df.iterrows():
        state_code = infra_row['STATE_CODE']
        district_code = infra_row['DISTIRCT_CODE']
        block_code = infra_row['BLOCK_CODE']
        block_quality = infra_row['_quality_score']
        
        # Generate Block-level SLA first
        block_admin_code = block_code
        block_nw_code = block_code  # Same as block_code for simplicity
        
        # Block SLA influenced by infrastructure quality
        block_base_sla = sla_baseline_mean + (block_quality - 0.7) * 30 * causal_strength
        
        block_sla_values = []
        for day in range(days_in_month):
            daily_variance = np.random.normal(0, sla_variance * (1 + (1-block_quality)*0.5))
            daily_sla = block_base_sla + daily_variance
            
            # Add outliers
            if np.random.random() < outlier_probability:
                daily_sla = np.random.uniform(0, 30)  # Extreme outage
            
            daily_sla = np.clip(daily_sla, 0, 100)
            block_sla_values.append(round(daily_sla, 2))
        
        # Create block SLA record
        block_sla_record = {
            'STATE_CODE': state_code,
            'DISTIRCT_CODE': district_code,
            'BLOCK_ADMIN_CODE': block_admin_code,
            'BLOCK_NW_CODE': block_nw_code,
            'GP_CODE': '',
            'STATE': infra_row['STATE'],
            'DISTRICT': infra_row['DISTRICT'],
            'BLOCK_ADMIN': infra_row['BLOCK'],
            'BLOCK_NW': infra_row['BLOCK'],
            'GP': '',
            'NODE_TYPE': 'Block'
        }
        
        for i, date_col in enumerate(date_columns):
            block_sla_record[date_col] = block_sla_values[i]
        
        sla_data.append(block_sla_record)
        
        # Generate GP-level SLAs under this block
        num_gps = max(10, int(np.random.normal(gps_per_block, gps_per_block*0.3)))
        
        for gp_idx in range(num_gps):
            gp_code = gp_code_counter
            gp_code_counter += 1
            gp_name = generate_place_name("GP", gp_code)
            
            # IMPORTANT: For 30% of GPs, admin and network blocks are different
            # This reflects real-world scenarios where administrative and network
            # boundaries don't always align due to historical reasons, infrastructure
            # constraints, or administrative reorganization
            if np.random.random() < admin_nw_different_pct:
                # Create a different network block for this GP
                different_nw_block = np.random.choice(infra_df['BLOCK_CODE'].values)
                gp_block_nw_code = different_nw_block
                gp_block_nw_name = infra_df[infra_df['BLOCK_CODE'] == different_nw_block]['BLOCK'].iloc[0]
            else:
                gp_block_nw_code = block_code
                gp_block_nw_name = infra_row['BLOCK']
            
            # GP SLA influenced by block SLA with additional variance
            gp_sla_values = []
            for day in range(days_in_month):
                # GP SLA correlated with block SLA but with additional local factors
                correlation_factor = causal_strength * 0.8
                local_factor = (1 - correlation_factor)
                
                gp_base = block_sla_values[day] * correlation_factor + sla_baseline_mean * local_factor
                gp_variance = np.random.normal(0, sla_variance * 0.8)
                gp_sla = gp_base + gp_variance
                
                # GP-specific outliers
                if np.random.random() < outlier_probability * 1.5:  # GPs more prone to outages
                    gp_sla = np.random.uniform(0, 40)
                
                gp_sla = np.clip(gp_sla, 0, 100)
                gp_sla_values.append(round(gp_sla, 2))
            
            gp_sla_record = {
                'STATE_CODE': state_code,
                'DISTIRCT_CODE': district_code,
                'BLOCK_ADMIN_CODE': block_admin_code,
                'BLOCK_NW_CODE': gp_block_nw_code,
                'GP_CODE': gp_code,
                'STATE': infra_row['STATE'],
                'DISTRICT': infra_row['DISTRICT'],
                'BLOCK_ADMIN': infra_row['BLOCK'],
                'BLOCK_NW': gp_block_nw_name,
                'GP': gp_name,
                'NODE_TYPE': 'GP'
            }
            
            for i, date_col in enumerate(date_columns):
                gp_sla_record[date_col] = gp_sla_values[i]
            
            sla_data.append(gp_sla_record)
            
            # Store GP info for FTTH generation
            avg_gp_sla = np.mean(gp_sla_values)
            gp_hierarchy[gp_code] = {
                'state_code': state_code,
                'district_code': district_code,
                'block_admin_code': block_admin_code,
                'block_nw_code': gp_block_nw_code,
                'state_name': infra_row['STATE'],
                'district_name': infra_row['DISTRICT'],
                'block_admin_name': infra_row['BLOCK'],
                'block_nw_name': gp_block_nw_name,
                'gp_name': gp_name,
                'avg_sla': avg_gp_sla
            }
    
    sla_df = pd.DataFrame(sla_data)
    
    # ===========================================
    # 3. GENERATE FTTH DATA (ftth_df)
    # ===========================================
    
    if progress_callback:
        progress_callback(0.7, "Generating FTTH connection data...")
    
    ftth_data = []
    
    # May 2025 date range - the report month
    report_month_start = datetime(2025, 5, 1)
    report_month_end = datetime(2025, 5, 31)
    
    customer_id_counter = 1
    village_code_counter = 1
    
    for gp_code, gp_info in gp_hierarchy.items():
        # Generate villages under this GP
        num_villages = max(2, int(np.random.normal(villages_per_gp, 1)))
        
        for village_idx in range(num_villages):
            village_code = village_code_counter
            village_code_counter += 1
            village_name = generate_place_name("Village", village_code)
            
            # Generate FTTH connections for this village
            num_connections = max(5, int(np.random.normal(connections_per_village, connections_per_village*0.4)))
            
            for conn_idx in range(num_connections):
                customer_id = f"CUST{customer_id_counter:08d}"
                customer_id_counter += 1
                
                # Determine connection scenario for May 2025 report
                scenario = np.random.choice([
                    'active_whole_month',    # Connected before May, active throughout May
                    'connected_in_may',      # Connected during May 2025
                    'disconnected_in_may'    # Disconnected during May 2025
                ], p=[0.7, 0.15, 0.15])
                
                if scenario == 'active_whole_month':
                    # Connected before May, active throughout May
                    connection_date = report_month_start - timedelta(days=np.random.randint(30, 365))
                    disconnection_date = None
                    days_active_in_may = 31
                    current_status = 'Active'
                    
                elif scenario == 'connected_in_may':
                    # Connected during May 2025
                    connection_day = np.random.randint(1, 32)  # Day 1-31 of May
                    connection_date = datetime(2025, 5, connection_day)
                    disconnection_date = None
                    days_active_in_may = 31 - connection_day + 1
                    current_status = 'Active'
                    
                else:  # disconnected_in_may
                    # Connected before May, disconnected during May
                    connection_date = report_month_start - timedelta(days=np.random.randint(30, 365))
                    disconnection_day = np.random.randint(1, 32)  # Day 1-31 of May
                    disconnection_date = datetime(2025, 5, disconnection_day)
                    days_active_in_may = disconnection_day
                    current_status = 'Disconnected'
                
                # User type
                user_type = np.random.choice(['Residential', 'Business', 'Government'], 
                                           p=[0.8, 0.18, 0.02])
                
                # Bandwidth (higher for business)
                if bandwidth_options:  # Check if bandwidth options exist
                    # Create probability distributions based on available options
                    num_options = len(bandwidth_options)
                    
                    if user_type == 'Business':
                        # Business users prefer higher bandwidth
                        probs = np.linspace(0.1, 0.3, num_options)
                        probs = probs / probs.sum()  # Normalize
                        bandwidth = np.random.choice(bandwidth_options, p=probs)
                    elif user_type == 'Government':
                        # Government users also prefer higher bandwidth
                        probs = np.linspace(0.15, 0.35, num_options)
                        probs = probs / probs.sum()  # Normalize
                        bandwidth = np.random.choice(bandwidth_options, p=probs)
                    else:  # Residential
                        # Residential users prefer lower bandwidth
                        probs = np.linspace(0.4, 0.05, num_options)
                        probs = probs / probs.sum()  # Normalize
                        bandwidth = np.random.choice(bandwidth_options, p=probs)
                else:
                    # Default bandwidth if no options selected
                    bandwidth = 100
                
                # Calculate May 2025 data utilization (30-100 GB range)
                # Base utilization influenced by SLA quality
                sla_impact = (gp_info['avg_sla'] / 100) * causal_strength + (1 - causal_strength) * 0.8
                
                # User type base utilization
                if user_type == 'Business':
                    base_utilization = np.random.uniform(60, 90)  # Higher base for business
                elif user_type == 'Government':
                    base_utilization = np.random.uniform(50, 80)
                else:
                    base_utilization = np.random.uniform(30, 70)  # Residential
                
                # Days active causation - more days = higher utilization potential
                days_factor = (days_active_in_may / 31) * days_active_causation + (1 - days_active_causation)
                
                # Final utilization calculation
                month_data_utilization = base_utilization * sla_impact * days_factor
                
                # Add some random variance
                month_data_utilization *= np.random.uniform(0.8, 1.2)
                
                # Ensure within 30-100 GB range
                month_data_utilization = np.clip(month_data_utilization, 30, 100)
                
                # Add outliers occasionally
                if np.random.random() < outlier_probability:
                    month_data_utilization = np.random.uniform(100, 150)  # Heavy users
                
                # If disconnected early in the month, reduce utilization significantly
                if current_status == 'Disconnected' and days_active_in_may < 10:
                    month_data_utilization *= 0.3  # Much lower utilization
                
                month_data_utilization = round(month_data_utilization, 2)
                
                ftth_record = {
                    'STATE_CODE': gp_info['state_code'],
                    'DISTIRCT_CODE': gp_info['district_code'],
                    'BLOCK_ADMIN_CODE': gp_info['block_admin_code'],
                    'BLOCK_NW_CODE': gp_info['block_nw_code'],
                    'GP_CODE': gp_code,
                    'VILLAGE_CODE': village_code,
                    'STATE': gp_info['state_name'],
                    'DISTRICT': gp_info['district_name'],
                    'BLOCK_ADMIN': gp_info['block_admin_name'],
                    'BLOCK_NW': gp_info['block_nw_name'],
                    'GP': gp_info['gp_name'],
                    'VILLAGE': village_name,
                    'CUSTOMER_ID': customer_id,
                    'CONNECTION_DATE': connection_date.strftime('%Y-%m-%d'),
                    'DISCONNECTION_DATE': disconnection_date.strftime('%Y-%m-%d') if disconnection_date else '',
                    'USER_TYPE': user_type,
                    'CURRENT_STATUS': current_status,
                    'BANDWIDTH_PROVIDED': bandwidth,
                    'MONTH_DATA_UTILISATION': month_data_utilization
                }
                
                ftth_data.append(ftth_record)
    
    ftth_df = pd.DataFrame(ftth_data)
    
    # ===========================================
    # CLEAN UP AND SORT
    # ===========================================
    
    if progress_callback:
        progress_callback(0.9, "Finalizing data...")
    
    # Remove internal columns
    infra_df = infra_df.drop(columns=['_quality_score'])
    
    # Convert data types to ensure compatibility
    # For SLA dataframe, keep GP_CODE as string (it contains empty strings for blocks)
    sla_df['GP_CODE'] = sla_df['GP_CODE'].astype(str)
    
    # Ensure numeric columns are properly typed
    infra_df['STATE_CODE'] = infra_df['STATE_CODE'].astype(int)
    infra_df['DISTIRCT_CODE'] = infra_df['DISTIRCT_CODE'].astype(int)
    infra_df['BLOCK_CODE'] = infra_df['BLOCK_CODE'].astype(int)
    infra_df['BATTERY_CAPACITY'] = infra_df['BATTERY_CAPACITY'].astype(float)
    
    sla_df['STATE_CODE'] = sla_df['STATE_CODE'].astype(int)
    sla_df['DISTIRCT_CODE'] = sla_df['DISTIRCT_CODE'].astype(int)
    sla_df['BLOCK_ADMIN_CODE'] = sla_df['BLOCK_ADMIN_CODE'].astype(int)
    sla_df['BLOCK_NW_CODE'] = sla_df['BLOCK_NW_CODE'].astype(int)
    
    ftth_df['STATE_CODE'] = ftth_df['STATE_CODE'].astype(int)
    ftth_df['DISTIRCT_CODE'] = ftth_df['DISTIRCT_CODE'].astype(int)
    ftth_df['BLOCK_ADMIN_CODE'] = ftth_df['BLOCK_ADMIN_CODE'].astype(int)
    ftth_df['BLOCK_NW_CODE'] = ftth_df['BLOCK_NW_CODE'].astype(int)
    ftth_df['GP_CODE'] = ftth_df['GP_CODE'].astype(int)
    ftth_df['VILLAGE_CODE'] = ftth_df['VILLAGE_CODE'].astype(int)
    ftth_df['BANDWIDTH_PROVIDED'] = ftth_df['BANDWIDTH_PROVIDED'].astype(int)
    ftth_df['MONTH_DATA_UTILISATION'] = ftth_df['MONTH_DATA_UTILISATION'].astype(float)
    
    # Sort dataframes for better organization
    infra_df = infra_df.sort_values(['STATE_CODE', 'DISTIRCT_CODE', 'BLOCK_CODE'])
    sla_df = sla_df.sort_values(['STATE_CODE', 'DISTIRCT_CODE', 'BLOCK_ADMIN_CODE', 'GP_CODE'])
    ftth_df = ftth_df.sort_values(['STATE_CODE', 'DISTIRCT_CODE', 'GP_CODE', 'VILLAGE_CODE', 'CUSTOMER_ID'])
    
    # Reset indices
    infra_df = infra_df.reset_index(drop=True)
    sla_df = sla_df.reset_index(drop=True)
    ftth_df = ftth_df.reset_index(drop=True)
    
    if progress_callback:
        progress_callback(1.0, "Data generation complete!")
    
    return infra_df, sla_df, ftth_df

def create_visualizations(infra_df, sla_df, ftth_df):
    """Create visualizations for the generated data"""
    
    # 1. FTTH Data Utilization Distribution
    fig_ftth = px.histogram(
        ftth_df, 
        x='MONTH_DATA_UTILISATION', 
        nbins=30,
        title='FTTH Data Utilization Distribution (May 2025)',
        labels={'MONTH_DATA_UTILISATION': 'Data Usage (GB)', 'count': 'Number of Connections'},
        color_discrete_sequence=['#1f77b4']
    )
    fig_ftth.update_layout(height=400)
    
    # 2. SLA Heatmap (average SLA by state and node type)
    date_cols = [col for col in sla_df.columns if any(month in col for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
    if date_cols:  # Only proceed if date columns exist
        sla_df_copy = sla_df.copy()
        sla_df_copy['avg_sla'] = sla_df_copy[date_cols].mean(axis=1)
        sla_summary = sla_df_copy.groupby(['STATE', 'NODE_TYPE'])['avg_sla'].mean().reset_index()
        sla_pivot = sla_summary.pivot(index='STATE', columns='NODE_TYPE', values='avg_sla')
        
        fig_sla = px.imshow(
            sla_pivot.values,
            x=sla_pivot.columns,
            y=sla_pivot.index,
            color_continuous_scale='RdYlGn',
            title='Average SLA Performance by State and Node Type',
            labels={'color': 'Average SLA %'}
        )
        fig_sla.update_layout(height=400)
    else:
        # Create empty figure if no date columns
        fig_sla = go.Figure()
        fig_sla.add_annotation(text="No SLA data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_sla.update_layout(height=400)
    
    # 3. Infrastructure Quality Score
    infra_df['infra_score'] = (
        infra_df['POWER_WORKING'].astype(int) * 0.3 +
        infra_df['AC_WORKING'].astype(int) * 0.2 +
        infra_df['GENERATOR_WORKING'].astype(int) * 0.2 +
        infra_df['MANPOWER_PRESENCE'].astype(int) * 0.2 +
        (infra_df['BATTERY_CAPACITY'] / 8) * 0.1
    )
    
    fig_infra = px.box(
        infra_df, 
        x='STATE', 
        y='infra_score',
        title='Infrastructure Quality Score by State',
        labels={'infra_score': 'Infrastructure Score (0-1)', 'STATE': 'State'}
    )
    fig_infra.update_xaxes(tickangle=45)
    fig_infra.update_layout(height=400)
    
    # 4. User Type vs Data Usage
    fig_user = px.box(
        ftth_df, 
        x='USER_TYPE', 
        y='MONTH_DATA_UTILISATION',
        title='Data Usage by User Type',
        labels={'MONTH_DATA_UTILISATION': 'Data Usage (GB)', 'USER_TYPE': 'User Type'}
    )
    fig_user.update_layout(height=400)
    
    return fig_ftth, fig_sla, fig_infra, fig_user

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv(index=False).encode('utf-8')

def main():
    st.title("📡 Telecom Synthetic Data Generator")
    st.markdown("Generate realistic synthetic telecom infrastructure, SLA, and FTTH data with configurable parameters and causal relationships.")
    
    # Sidebar for parameters
    st.sidebar.header("📊 Configuration Parameters")
    
    # Geographic Parameters
    st.sidebar.subheader("🗺️ Geographic Scope")
    num_states = st.sidebar.slider("Number of States", 3, 10, 7)
    districts_per_state = st.sidebar.slider("Districts per State (avg)", 3, 10, 6)
    blocks_per_district = st.sidebar.slider("Blocks per District (avg)", 5, 15, 10)
    gps_per_block = st.sidebar.slider("GPs per Block (avg)", 10, 30, 20)
    villages_per_gp = st.sidebar.slider("Villages per GP (avg)", 2, 5, 3)
    connections_per_village = st.sidebar.slider("FTTH Connections per Village (avg)", 5, 20, 10)
    
    # Variability Parameters
    st.sidebar.subheader("🎛️ Variability & Causality")
    infra_quality_variance = st.sidebar.slider("Infrastructure Quality Variance", 0.1, 0.5, 0.3, 0.05)
    sla_baseline_mean = st.sidebar.slider("SLA Baseline Mean (%)", 70, 95, 85)
    sla_variance = st.sidebar.slider("SLA Standard Deviation", 5, 25, 15)
    causal_strength = st.sidebar.slider("Causal Relationship Strength", 0.0, 1.0, 0.5, 0.1)
    outlier_probability = st.sidebar.slider("Outlier Probability", 0.01, 0.15, 0.05, 0.01)
    days_active_causation = st.sidebar.slider("Days-Active Impact on Usage", 0.0, 1.0, 0.7, 0.1)
    admin_nw_different_pct = st.sidebar.slider("% GPs with Different Admin/NW Blocks", 0.0, 0.5, 0.3, 0.05)
    
    # FTTH Parameters
    st.sidebar.subheader("🏠 FTTH Configuration")
    bandwidth_options = st.sidebar.multiselect(
        "Available Bandwidth Options (Mbps)", 
        [25, 50, 100, 200, 500, 1000], 
        default=[50, 100, 200, 500, 1000]
    )
    disconnection_rate = st.sidebar.slider("Monthly Disconnection Rate", 0.05, 0.30, 0.15, 0.05)
    
    # Temporal Parameters
    st.sidebar.subheader("📅 Temporal Settings")
    sla_month = st.sidebar.selectbox("Report Month", ["2025-05", "2025-04", "2025-06"], index=0)
    
    # Generate Data Button
    if st.sidebar.button("🚀 Generate Data", type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Generate data
            infra_df, sla_df, ftth_df = generate_telecom_synthetic_data(
                num_states=num_states,
                districts_per_state=districts_per_state,
                blocks_per_district=blocks_per_district,
                gps_per_block=gps_per_block,
                villages_per_gp=villages_per_gp,
                connections_per_village=connections_per_village,
                infra_quality_variance=infra_quality_variance,
                sla_baseline_mean=sla_baseline_mean,
                sla_variance=sla_variance,
                causal_strength=causal_strength,
                outlier_probability=outlier_probability,
                days_active_causation=days_active_causation,
                admin_nw_different_pct=admin_nw_different_pct,
                sla_month=sla_month,
                bandwidth_options=bandwidth_options,
                disconnection_rate=disconnection_rate,
                progress_callback=update_progress
            )
            
            # Store in session state
            st.session_state.infra_df = infra_df
            st.session_state.sla_df = sla_df
            st.session_state.ftth_df = ftth_df
            st.session_state.generated = True
            
            status_text.text("✅ Data generation completed successfully!")
            progress_bar.empty()
            
        except Exception as e:
            st.error(f"Error generating data: {str(e)}")
            return
    
    # Display results if data has been generated
    if hasattr(st.session_state, 'generated') and st.session_state.generated:
        
        infra_df = st.session_state.infra_df
        sla_df = st.session_state.sla_df
        ftth_df = st.session_state.ftth_df
        
        # Summary metrics
        st.header("📊 Data Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total States", infra_df['STATE'].nunique())
            st.metric("Total Districts", infra_df['DISTRICT'].nunique())
        
        with col2:
            st.metric("Total Blocks", infra_df['BLOCK'].nunique())
            st.metric("Total GPs", sla_df[sla_df['NODE_TYPE'] == 'GP']['GP_CODE'].nunique())
        
        with col3:
            st.metric("Total Villages", ftth_df['VILLAGE_CODE'].nunique())
            st.metric("Total Connections", len(ftth_df))
        
        with col4:
            st.metric("Active Connections", len(ftth_df[ftth_df['CURRENT_STATUS'] == 'Active']))
            st.metric("Avg Data Usage", f"{ftth_df['MONTH_DATA_UTILISATION'].mean():.1f} GB")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Visualizations", "🏗️ Infrastructure Data", "📡 SLA Data", "🏠 FTTH Data"])
        
        with tab1:
            st.header("📈 Data Visualizations")
            
            # Create visualizations
            fig_ftth, fig_sla, fig_infra, fig_user = create_visualizations(infra_df, sla_df, ftth_df)
            
            # Display in 2x2 grid
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_ftth, use_container_width=True)
                st.plotly_chart(fig_infra, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_sla, use_container_width=True)
                st.plotly_chart(fig_user, use_container_width=True)
            
            # Additional insights
            st.subheader("🔍 Key Insights")
            
            # Calculate correlations
            # Infrastructure score vs average SLA
            infra_summary = infra_df.groupby('BLOCK_CODE').first()
            infra_summary['infra_score'] = (
                infra_summary['POWER_WORKING'].astype(int) * 0.3 +
                infra_summary['AC_WORKING'].astype(int) * 0.2 +
                infra_summary['GENERATOR_WORKING'].astype(int) * 0.2 +
                infra_summary['MANPOWER_PRESENCE'].astype(int) * 0.2 +
                (infra_summary['BATTERY_CAPACITY'] / 8) * 0.1
            )
            
            # Get block-level SLA averages
            date_cols = [col for col in sla_df.columns if any(month in col for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
            if date_cols:
                block_sla = sla_df[sla_df['NODE_TYPE'] == 'Block'].copy()
                block_sla['avg_sla'] = block_sla[date_cols].mean(axis=1)
            
            # Merge for correlation
            merged_data = pd.merge(
                infra_summary[['infra_score']].reset_index(),
                block_sla[['BLOCK_ADMIN_CODE', 'avg_sla']],
                left_on='BLOCK_CODE',
                right_on='BLOCK_ADMIN_CODE'
            )
            
            correlation = merged_data['infra_score'].corr(merged_data['avg_sla'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Infrastructure-SLA Correlation:** {correlation:.3f}")
            
            with col2:
                avg_sla_all = sla_df[date_cols].mean().mean()
                st.info(f"**Overall Average SLA:** {avg_sla_all:.1f}%")
            
            with col3:
                pct_power_working = (infra_df['POWER_WORKING'].sum() / len(infra_df)) * 100
                st.info(f"**Blocks with Working Power:** {pct_power_working:.1f}%")
        
        with tab2:
            st.header("🏗️ Infrastructure Data")
            
            # Filters
            col1, col2 = st.columns(2)
            
            with col1:
                selected_states = st.multiselect(
                    "Filter by State",
                    options=infra_df['STATE'].unique(),
                    default=infra_df['STATE'].unique()
                )
            
            with col2:
                power_filter = st.checkbox("Show only blocks with working power", value=False)
            
            # Apply filters
            filtered_infra = infra_df[infra_df['STATE'].isin(selected_states)]
            if power_filter:
                filtered_infra = filtered_infra[filtered_infra['POWER_WORKING'] == True]
            
            # Display data
            st.dataframe(filtered_infra, use_container_width=True)
            
            # Download button with custom filename
            csv = convert_df_to_csv(filtered_infra)
            col1, col2 = st.columns([3, 1])
            with col1:
                download_keyword_infra = st.text_input(
                    "Filename keyword", 
                    key="infra_keyword",
                    placeholder="optional keyword"
                )
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename_suffix = f"{download_keyword_infra}_{timestamp}" if download_keyword_infra else timestamp
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"infrastructure_data_{filename_suffix}.csv",
                    mime="text/csv"
                )
        
        with tab3:
            st.header("📡 SLA Data")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                node_type_filter = st.selectbox(
                    "Node Type",
                    options=['All', 'Block', 'GP'],
                    index=0
                )
            
            with col2:
                sla_threshold = st.slider(
                    "Minimum Average SLA %",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=5
                )
            
            with col3:
                selected_date = st.selectbox(
                    "Select Date",
                    options=date_cols,
                    index=0
                )
            
            # Apply filters
            filtered_sla = sla_df.copy()
            
            if node_type_filter != 'All':
                filtered_sla = filtered_sla[filtered_sla['NODE_TYPE'] == node_type_filter]
            
            # Calculate average SLA for filtering
            filtered_sla['avg_sla'] = filtered_sla[date_cols].mean(axis=1)
            filtered_sla = filtered_sla[filtered_sla['avg_sla'] >= sla_threshold]
            
            # Show summary statistics
            st.subheader("SLA Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average SLA", f"{filtered_sla[selected_date].mean():.1f}%")
            
            with col2:
                st.metric("Minimum SLA", f"{filtered_sla[selected_date].min():.1f}%")
            
            with col3:
                st.metric("Maximum SLA", f"{filtered_sla[selected_date].max():.1f}%")
            
            # Display data (without avg_sla column for cleaner view)
            display_cols = [col for col in filtered_sla.columns if col != 'avg_sla']
            st.dataframe(filtered_sla[display_cols], use_container_width=True)
            
            # Download button with custom filename
            csv = convert_df_to_csv(filtered_sla)
            col1, col2 = st.columns([3, 1])
            with col1:
                download_keyword_sla = st.text_input(
                    "Filename keyword", 
                    key="sla_keyword",
                    placeholder="optional keyword"
                )
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename_suffix = f"{download_keyword_sla}_{timestamp}" if download_keyword_sla else timestamp
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"sla_data_{filename_suffix}.csv",
                    mime="text/csv"
                )
        
        with tab4:
            st.header("🏠 FTTH Data")
            
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                user_type_filter = st.multiselect(
                    "User Type",
                    options=ftth_df['USER_TYPE'].unique(),
                    default=ftth_df['USER_TYPE'].unique()
                )
            
            with col2:
                status_filter = st.selectbox(
                    "Connection Status",
                    options=['All', 'Active', 'Disconnected'],
                    index=0
                )
            
            with col3:
                bandwidth_filter = st.multiselect(
                    "Bandwidth (Mbps)",
                    options=sorted(ftth_df['BANDWIDTH_PROVIDED'].unique()),
                    default=sorted(ftth_df['BANDWIDTH_PROVIDED'].unique())
                )
            
            with col4:
                usage_range = st.slider(
                    "Data Usage Range (GB)",
                    min_value=int(ftth_df['MONTH_DATA_UTILISATION'].min()),
                    max_value=int(ftth_df['MONTH_DATA_UTILISATION'].max()),
                    value=(int(ftth_df['MONTH_DATA_UTILISATION'].min()), 
                           int(ftth_df['MONTH_DATA_UTILISATION'].max()))
                )
            
            # Apply filters
            filtered_ftth = ftth_df[
                (ftth_df['USER_TYPE'].isin(user_type_filter)) &
                (ftth_df['BANDWIDTH_PROVIDED'].isin(bandwidth_filter)) &
                (ftth_df['MONTH_DATA_UTILISATION'] >= usage_range[0]) &
                (ftth_df['MONTH_DATA_UTILISATION'] <= usage_range[1])
            ]
            
            if status_filter != 'All':
                filtered_ftth = filtered_ftth[filtered_ftth['CURRENT_STATUS'] == status_filter]
            
            # Summary metrics for filtered data
            st.subheader("Filtered Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Connections", len(filtered_ftth))
            
            with col2:
                st.metric("Active Connections", len(filtered_ftth[filtered_ftth['CURRENT_STATUS'] == 'Active']))
            
            with col3:
                st.metric("Avg Data Usage", f"{filtered_ftth['MONTH_DATA_UTILISATION'].mean():.1f} GB")
            
            with col4:
                st.metric("Total Data Usage", f"{filtered_ftth['MONTH_DATA_UTILISATION'].sum():.0f} GB")
            
            # Display data
            st.dataframe(filtered_ftth, use_container_width=True)
            
            # Download button with custom filename
            csv = convert_df_to_csv(filtered_ftth)
            col1, col2 = st.columns([3, 1])
            with col1:
                download_keyword_ftth = st.text_input(
                    "Filename keyword", 
                    key="ftth_keyword",
                    placeholder="optional keyword"
                )
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename_suffix = f"{download_keyword_ftth}_{timestamp}" if download_keyword_ftth else timestamp
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv,
                    file_name=f"ftth_data_{filename_suffix}.csv",
                    mime="text/csv"
                )
        
        # Download All Data Section
        st.header("💾 Download All Data")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Custom keyword input
            download_keyword = st.text_input(
                "Enter a keyword for filenames (optional)",
                placeholder="e.g., 'test1', 'final', 'v2'",
                help="This keyword will be added to all downloaded filenames to prevent overwriting"
            )
        
        with col2:
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create filename suffix
            if download_keyword:
                filename_suffix = f"{download_keyword}_{timestamp}"
            else:
                filename_suffix = timestamp
            
            st.info(f"Files will be saved as: *_data_{filename_suffix}.csv")
        
        with col3:
            # Download all button
            if st.button("⬇️ Download All Data", type="primary", use_container_width=True):
                # Create a zip file with all three CSVs
                import zipfile
                import io
                
                # Create a BytesIO object to hold the zip file
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add infrastructure data
                    infra_csv = infra_df.to_csv(index=False)
                    zip_file.writestr(f"infrastructure_data_{filename_suffix}.csv", infra_csv)
                    
                    # Add SLA data
                    sla_csv = sla_df.to_csv(index=False)
                    zip_file.writestr(f"sla_data_{filename_suffix}.csv", sla_csv)
                    
                    # Add FTTH data
                    ftth_csv = ftth_df.to_csv(index=False)
                    zip_file.writestr(f"ftth_data_{filename_suffix}.csv", ftth_csv)
                
                # Prepare the zip file for download
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📦 Download ZIP Archive",
                    data=zip_buffer.getvalue(),
                    file_name=f"telecom_data_{filename_suffix}.zip",
                    mime="application/zip"
                )
        
        # Individual download options
        with st.expander("📄 Download Individual Files"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Infrastructure Data")
                csv = convert_df_to_csv(infra_df)
                st.download_button(
                    label="⬇️ Download Infrastructure CSV",
                    data=csv,
                    file_name=f"infrastructure_data_{filename_suffix}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.subheader("SLA Data")
                csv = convert_df_to_csv(sla_df)
                st.download_button(
                    label="⬇️ Download SLA CSV",
                    data=csv,
                    file_name=f"sla_data_{filename_suffix}.csv",
                    mime="text/csv"
                )
            
            with col3:
                st.subheader("FTTH Data")
                csv = convert_df_to_csv(ftth_df)
                st.download_button(
                    label="⬇️ Download FTTH CSV",
                    data=csv,
                    file_name=f"ftth_data_{filename_suffix}.csv",
                    mime="text/csv"
                )
        
        # Information about the data
        with st.expander("ℹ️ About this Data"):
            st.markdown("""
            ### Synthetic Data Generation Details
            
            This synthetic telecom dataset simulates realistic relationships between:
            
            **Infrastructure Quality** → **SLA Performance** → **Data Utilization**
            
            #### Key Features:
            
            1. **Causal Relationships**: Infrastructure quality influences SLA performance, which in turn affects customer data usage
            2. **Realistic Variance**: Natural variations and outliers are included to simulate real-world conditions
            3. **Hierarchical Structure**: State → District → Block → GP → Village → Customer
            4. **Temporal Patterns**: Daily SLA variations and monthly data usage patterns
            
            #### Configurable Parameters:
            
            - **Geographic Scope**: Control the size of the generated dataset
            - **Variability**: Adjust how much variation exists in the data
            - **Causal Strength**: Control how strongly infrastructure affects SLA and usage
            - **Outliers**: Include realistic edge cases and anomalies
            
            #### Use Cases:
            
            - Testing analytics dashboards and reports
            - Training machine learning models
            - Demonstrating data relationships
            - Performance testing with various data sizes
            """)
    
    else:
        # Show instructions when no data is generated
        st.info("👈 Configure parameters in the sidebar and click **Generate Data** to create synthetic telecom data.")
        
        st.markdown("""
        ### Welcome to the Telecom Synthetic Data Generator!
        
        This tool generates realistic synthetic data for telecom infrastructure, including:
        
        1. **Infrastructure Data**: Block-level infrastructure information (power, AC, generator, battery, etc.)
        2. **SLA Data**: Daily Service Level Agreement performance for blocks and GPs
        3. **FTTH Data**: Fiber-to-the-Home customer connections and usage data
        
        #### Getting Started:
        
        1. Adjust the parameters in the sidebar to configure your dataset
        2. Click the **Generate Data** button
        3. Explore the generated data through visualizations and tables
        4. Download the data as CSV files for further analysis
        
        #### Key Features:
        
        - **Configurable Causal Relationships**: Infrastructure quality affects SLA, which impacts data usage
        - **Realistic Variations**: Includes natural variance and outliers
        - **Hierarchical Geographic Structure**: From states down to individual customers
        - **Temporal Patterns**: Daily SLA variations for the selected month
        """)

if __name__ == "__main__":
    main()