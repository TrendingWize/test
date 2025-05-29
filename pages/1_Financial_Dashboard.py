# pages/1_Financial_Dashboard.py
import streamlit as st
# from components.navigation import top_navigation_bar # Import if you use streamlit-option-menu approach
from components.company_profile import company_profile_tab_content
from components.income_statement import income_statement_tab_content
from components.balance_sheet import balance_sheet_tab_content
from components.cash_flow import cash_flow_tab_content # Assuming you'll create this

st.set_page_config(
    page_title="Financial Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed"
)


# --- Display Custom Top Navigation (Visual Consistency) ---
st.markdown("""
<style>
    .top-nav-container { background-color: #f0f2f6; padding: 10px 0px; border-bottom: 1px solid #cccccc; margin-bottom: 20px; text-align: center; }
    .top-nav-container a { margin: 0 15px; text-decoration: none; color: #333; font-weight: 500; font-size: 1.1em; }
    .top-nav-container a:hover { color: #007bff; }
</style>
<div class="top-nav-container">
    <a href="/" target="_self">Home</a>
    <a href="/Financial_Dashboard" target="_self">Financial Dashboard</a>
    <a href="/Contact_Us" target="_self">Contact Us</a>
</div>
""", unsafe_allow_html=True)
# selected_page = top_navigation_bar() # If using streamlit-option-menu


# --- Main Title for this page ---
st.title("📊 Financial Statements Dashboard")
# st.markdown("Analyze key financial statements using data from Neo4j.") # Subtitle already on home

# --- Initialize selected_symbol in session state ---
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = "AAPL"

# --- Search Box and Button ---
with st.container():
    st.markdown("<div class='search-container'>", unsafe_allow_html=True)
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        current_input_symbol = st.text_input(
            "Enter Company Symbol (e.g., AAPL, NVDA):", 
            value=st.session_state.selected_symbol,
            key="symbol_search_input_dashboard_page", # Unique key for this page
            label_visibility="collapsed",
            placeholder="Enter Symbol (e.g., AAPL)"
        ).upper()
    with search_col2:
        if st.button("Search", key="search_button_dashboard_page", use_container_width=True):
            if current_input_symbol:
                st.session_state.selected_symbol = current_input_symbol
            else:
                st.warning("Please enter a symbol.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# --- Tabs for Financial Statements ---
symbol_to_display = st.session_state.selected_symbol

tab_profile, tab_is, tab_bs, tab_cf = st.tabs([
    "👤 Company Profile", 
    "💰 Income Statement", 
    "🏦 Balance Sheet", 
    "🌊 Cash Flow Statement"
])

with tab_profile:
    if symbol_to_display: company_profile_tab_content(symbol_to_display)
    else: st.info("Please enter a symbol and click Search.")
with tab_is:
    if symbol_to_display: income_statement_tab_content(symbol_to_display)
    else: st.info("Please enter a symbol and click Search.")
with tab_bs:
    if symbol_to_display: balance_sheet_tab_content(symbol_to_display)
    else: st.info("Please enter a symbol and click Search.")
with tab_cf:
    if symbol_to_display:
        cash_flow_tab_content(symbol_to_display)
    else:
        st.info("Please enter a company symbol in the sidebar.")