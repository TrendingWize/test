import streamlit as st
st.set_page_config(page_title="Financial Insights – Home", layout="wide", page_icon="🏠")

from components.nav_bar import render_nav_bar     # ← navigation bar
from styles import load_global_css 

render_nav_bar()
load_global_css()    
# optional: hide sidebar completely
st.markdown("""
<style>
section[data-testid="stSidebar"]{display:none!important;}
button[data-testid="baseButton-header"]{display:none!important;}
</style>
""", unsafe_allow_html=True)

# --- Load Global CSS ---
# Apply custom styles across the entire application.
load_global_css()

# --- Initialize Global Session State ---
# This ensures that a default symbol is set when the app first loads.
# Other pages can then rely on this session state variable.
if 'global_selected_symbol' not in st.session_state:
    st.session_state.global_selected_symbol = "AAPL" # Default company symbol

# --- Custom Top Navigation Bar ---
# This function (from components/navigation.py) renders the main navigation links.
#R_custom_navigation()

# --- Landing Page Content ---

# Main title of the landing page
st.title("Welcome to Financial Insights!")

# Subheader providing a brief overview of the application
st.subheader("Your one-stop platform for financial statement analysis, company profiles, and AI-driven insights.")

# Markdown block detailing the key features of the application
st.markdown("""
    This application provides tools to explore:
    *   **Company Profiles:** Get a quick overview of publicly traded companies.
    *   **Financial Statements:** Dive deep into Income Statements, Balance Sheets, and Cash Flow Statements.
    *   **AI-Powered Analysis:** View comprehensive financial analysis reports generated by AI.
    *   **Interactive Data:** Visualize trends and analyze key metrics with interactive charts and tables.

    Navigate using the links above to get started.
""")

# Visual separator
st.markdown("---")

# Placeholder image for visual appeal - consider a more relevant image or a carousel of features.
# Image sourced from Unsplash, free to use.
st.image(
    "https://images.unsplash.com/photo-1554260570-e708275a248a?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80",
    caption="Data-Driven Financial Insights",
    use_column_width=True # Make image responsive to column width
)

# Another visual separator
st.markdown("---")

# Informational message guiding the user
st.info("Select a page from the top navigation bar to explore different sections of the application.")


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display:none !important;}
    [data-testid="collapsedControl"] {display:none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)
# You could add more sections to your landing page here, such as:
# - A brief "How to Use" guide.
# - "Featured Company" or "Market Movers" (if you have dynamic data).
# - Links to data sources or disclaimers.