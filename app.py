import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="⚡ Electricity Consumption Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0.5rem 1rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'weekly_data' not in st.session_state:
    st.session_state.weekly_data = []

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Main header
st.markdown('<h1 class="main-header">⚡ Electricity Consumption Calculator</h1>', unsafe_allow_html=True)

# Sidebar for user profile
with st.sidebar:
    st.header("👤 User Profile")
    
    name = st.text_input("Enter your name:", key="name")
    age = st.number_input("Enter your age:", min_value=1, max_value=120, value=25, key="age")
    area = st.text_input("Enter your area:", key="area")
    city = st.text_input("Enter your city:", key="city")
    
    st.subheader("🏠 Housing Details")
    house_type = st.selectbox(
        "Housing Type:",
        ["Flat", "Tenement"],
        key="house_type"
    )
    
    bhk_type = st.selectbox(
        "BHK Type:",
        ["1BHK", "2BHK", "3BHK"],
        key="bhk_type"
    )
    
    # Store user profile
    st.session_state.user_profile = {
        'name': name,
        'age': age,
        'area': area,
        'city': city,
        'house_type': house_type,
        'bhk_type': bhk_type
    }

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Weekly Calculator", "📈 Usage Analytics", "⚙️ Appliance Settings", "📋 Usage History"])

with tab1:
    st.header("Weekly Electricity Consumption Calculator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚡ Appliance Configuration")
        
        # Appliance wattage (user can modify)
        with st.expander("🔧 Appliance Power Settings (Watts)", expanded=True):
            light_watt = st.number_input("Light bulb power (W):", min_value=1, value=60, key="light_watt")
            fan_watt = st.number_input("Fan power (W):", min_value=1, value=75, key="fan_watt")
            ac_watt = st.number_input("AC power (W):", min_value=500, value=1500, key="ac_watt")
            fridge_watt = st.number_input("Refrigerator power (W):", min_value=100, value=300, key="fridge_watt")
            washing_machine_watt = st.number_input("Washing machine power (W):", min_value=300, value=500, key="washing_machine_watt")
        
        # Calculate base consumption based on BHK type
        bhk_lower = bhk_type.lower()
        if bhk_lower == '1bhk':
            default_lights = 2
            default_fans = 2
        elif bhk_lower == '2bhk':
            default_lights = 3
            default_fans = 3
        elif bhk_lower == '3bhk':
            default_lights = 4
            default_fans = 4
        else:
            default_lights = 2
            default_fans = 2
        
        # Base consumption
        base_consumption = (default_lights * light_watt) + (default_fans * fan_watt)
        
        st.subheader("🏠 Additional Appliances")
        
        # Additional appliances
        col_a, col_b = st.columns(2)
        
        with col_a:
            has_ac = st.checkbox("Air Conditioner", key="has_ac")
            num_ac = st.number_input("Number of ACs:", min_value=0, value=0 if not has_ac else 1, key="num_ac") if has_ac else 0
            
            has_fridge = st.checkbox("Refrigerator", key="has_fridge")
            num_fridge = st.number_input("Number of Refrigerators:", min_value=0, value=0 if not has_fridge else 1, key="num_fridge") if has_fridge else 0
        
        with col_b:
            has_washing_machine = st.checkbox("Washing Machine", key="has_washing_machine")
            num_washing_machine = st.number_input("Number of Washing Machines:", min_value=0, value=0 if not has_washing_machine else 1, key="num_washing_machine") if has_washing_machine else 0
        
        # Daily usage hours
        st.subheader("⏰ Daily Usage Hours")
        
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            light_hours = st.slider("Lights usage (hours/day):", 0, 24, 6, key="light_hours")
            fan_hours = st.slider("Fans usage (hours/day):", 0, 24, 8, key="fan_hours")
            ac_hours = st.slider("AC usage (hours/day):", 0, 24, 4 if has_ac else 0, key="ac_hours")
        
        with col_h2:
            fridge_hours = st.slider("Fridge usage (hours/day):", 0, 24, 24 if has_fridge else 0, key="fridge_hours")
            washing_machine_hours = st.slider("Washing machine usage (hours/day):", 0, 24, 1 if has_washing_machine else 0, key="washing_machine_hours")
    
    with col2:
        st.subheader("📊 Consumption Summary")
        
        # Calculate total daily consumption in kWh
        daily_consumption = (
            (default_lights * light_watt * light_hours) +
            (default_fans * fan_watt * fan_hours) +
            (num_ac * ac_watt * ac_hours) +
            (num_fridge * fridge_watt * fridge_hours) +
            (num_washing_machine * washing_machine_watt * washing_machine_hours)
        ) / 1000  # Convert to kWh
        
        weekly_consumption = daily_consumption * 7
        monthly_consumption = daily_consumption * 30
        
        # Display metrics
        st.metric("Daily Consumption", f"{daily_consumption:.2f} kWh")
        st.metric("Weekly Consumption", f"{weekly_consumption:.2f} kWh")
        st.metric("Monthly Consumption", f"{monthly_consumption:.2f} kWh")
        
        # Estimated cost (assuming ₹5 per kWh)
        cost_per_kwh = 5
        daily_cost = daily_consumption * cost_per_kwh
        weekly_cost = weekly_consumption * cost_per_kwh
        monthly_cost = monthly_consumption * cost_per_kwh
        
        st.subheader("💰 Estimated Cost")
        st.metric("Daily Cost", f"₹{daily_cost:.2f}")
        st.metric("Weekly Cost", f"₹{weekly_cost:.2f}")
        st.metric("Monthly Cost", f"₹{monthly_cost:.2f}")
        
        # Save weekly data button
        if st.button("💾 Save This Week's Data", type="primary"):
            week_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'daily_consumption': daily_consumption,
                'weekly_consumption': weekly_consumption,
                'appliances': {
                    'lights': default_lights,
                    'fans': default_fans,
                    'ac': num_ac,
                    'fridge': num_fridge,
                    'washing_machine': num_washing_machine
                },
                'usage_hours': {
                    'lights': light_hours,
                    'fans': fan_hours,
                    'ac': ac_hours,
                    'fridge': fridge_hours,
                    'washing_machine': washing_machine_hours
                }
            }
            st.session_state.weekly_data.append(week_data)
            st.success("Weekly data saved successfully! 🎉")

with tab2:
    st.header("📈 Usage Analytics")
    
    if st.session_state.weekly_data:
        # Create DataFrame from saved data
        df = pd.DataFrame(st.session_state.weekly_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weekly consumption trend
            fig_trend = px.line(
                df, 
                x='date', 
                y='weekly_consumption',
                title='Weekly Consumption Trend',
                labels={'weekly_consumption': 'Consumption (kWh)', 'date': 'Date'}
            )
            fig_trend.update_layout(height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            # Daily consumption comparison
            fig_daily = px.bar(
                df, 
                x='date', 
                y='daily_consumption',
                title='Daily Consumption Comparison',
                labels={'daily_consumption': 'Consumption (kWh)', 'date': 'Date'}
            )
            fig_daily.update_layout(height=400)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Appliance-wise breakdown for latest entry
        if len(df) > 0:
            latest_data = df.iloc[-1]
            
            # Create appliance breakdown
            appliance_data = []
            if latest_data['appliances']['lights'] > 0:
                consumption = latest_data['appliances']['lights'] * light_watt * latest_data['usage_hours']['lights'] / 1000
                appliance_data.append({'Appliance': 'Lights', 'Consumption': consumption})
            
            if latest_data['appliances']['fans'] > 0:
                consumption = latest_data['appliances']['fans'] * fan_watt * latest_data['usage_hours']['fans'] / 1000
                appliance_data.append({'Appliance': 'Fans', 'Consumption': consumption})
            
            if latest_data['appliances']['ac'] > 0:
                consumption = latest_data['appliances']['ac'] * ac_watt * latest_data['usage_hours']['ac'] / 1000
                appliance_data.append({'Appliance': 'Air Conditioner', 'Consumption': consumption})
            
            if latest_data['appliances']['fridge'] > 0:
                consumption = latest_data['appliances']['fridge'] * fridge_watt * latest_data['usage_hours']['fridge'] / 1000
                appliance_data.append({'Appliance': 'Refrigerator', 'Consumption': consumption})
            
            if latest_data['appliances']['washing_machine'] > 0:
                consumption = latest_data['appliances']['washing_machine'] * washing_machine_watt * latest_data['usage_hours']['washing_machine'] / 1000
                appliance_data.append({'Appliance': 'Washing Machine', 'Consumption': consumption})
            
            if appliance_data:
                appliance_df = pd.DataFrame(appliance_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart for appliance breakdown
                    fig_pie = px.pie(
                        appliance_df, 
                        values='Consumption', 
                        names='Appliance',
                        title='Latest Week - Appliance-wise Consumption'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart for appliance consumption
                    fig_bar = px.bar(
                        appliance_df, 
                        x='Appliance', 
                        y='Consumption',
                        title='Appliance-wise Daily Consumption',
                        labels={'Consumption': 'Consumption (kWh)'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    else:
        st.info("No data available yet. Please save some weekly data first! 📊")

with tab3:
    st.header("⚙️ Appliance Settings & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔌 Current Appliance Configuration")
        
        config_data = {
            'Appliance': ['Lights', 'Fans', 'AC', 'Refrigerator', 'Washing Machine'],
            'Power (W)': [light_watt, fan_watt, ac_watt, fridge_watt, washing_machine_watt],
            'Quantity': [default_lights, default_fans, num_ac, num_fridge, num_washing_machine],
            'Hours/Day': [light_hours, fan_hours, ac_hours, fridge_hours, washing_machine_hours]
        }
        
        config_df = pd.DataFrame(config_data)
        st.dataframe(config_df, use_container_width=True)
    
    with col2:
        st.subheader("💡 Energy Saving Tips")
        
        tips = [
            "💡 Use LED bulbs to reduce lighting consumption by 75%",
            "🌪️ Clean AC filters regularly for optimal efficiency",
            "❄️ Keep refrigerator at optimal temperature (37-38°F)",
            "👕 Wash clothes in cold water when possible",
            "🔌 Unplug electronics when not in use",
            "🌡️ Use ceiling fans to reduce AC usage",
            "⏰ Use timer functions on appliances",
            "🪟 Keep curtains closed during hot days"
        ]
        
        for tip in tips:
            st.write(tip)

with tab4:
    st.header("📋 Usage History")
    
    if st.session_state.weekly_data:
        # Display saved data in a table
        history_df = pd.DataFrame(st.session_state.weekly_data)
        
        # Format the display
        display_df = history_df[['date', 'daily_consumption', 'weekly_consumption']].copy()
        display_df.columns = ['Date', 'Daily Consumption (kWh)', 'Weekly Consumption (kWh)']
        display_df['Daily Cost (₹)'] = display_df['Daily Consumption (kWh)'] * 5
        display_df['Weekly Cost (₹)'] = display_df['Weekly Consumption (kWh)'] * 5
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button for data
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Usage History",
            data=csv,
            file_name=f"electricity_usage_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.weekly_data = []
            st.success("History cleared successfully!")
            st.rerun()
    
    else:
        st.info("No usage history available yet. Start tracking your weekly consumption! 📈")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Track your electricity usage and save energy! 🌱")