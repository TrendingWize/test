# pages/2_AI_Analysis.py
import streamlit as st
from components.ai_analysis import ai_analysis_tab_content # Import the content function

st.set_page_config(
    page_title="AI Financial Analysis",
    layout="wide",
    page_icon="🤖",
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
    <a href="/AI_Analysis" target="_self">AI Analysis</a>
    <a href="/Contact_Us" target="_self">Contact Us</a>
</div>
""", unsafe_allow_html=True)

# --- Initialize selected_symbol in session state if not present ---
# This allows this page to potentially have its own search or use a global one
if 'selected_symbol_aianalysis' not in st.session_state: # Use a page-specific key for its own search bar
    st.session_state.selected_symbol_aianalysis = "AAPL" # Default or could be synced with a global state

# --- Search Bar (Optional - if you want symbol selection specific to this page) ---
# If you want to use the SAME symbol selected on the Financial Dashboard page,
# you would need a more robust way to share st.session_state across pages or use query params.
# For now, let's give it its own search bar or assume it uses a default.
st.sidebar.subheader("AI Analysis Options") # Example: options in sidebar for this page
symbol_for_ai = st.sidebar.text_input(
    "Company Symbol for AI Analysis:",
    value=st.session_state.selected_symbol_aianalysis,
    key="ai_analysis_symbol_input"
).upper()

if st.sidebar.button("Load AI Analysis", key="load_ai_analysis_btn"):
    st.session_state.selected_symbol_aianalysis = symbol_for_ai
    # st.rerun() # Rerun if necessary, ai_analysis_tab_content will use the new symbol

# --- AI Analysis Content ---
if st.session_state.selected_symbol_aianalysis:
    # The ai_analysis_tab_content function already handles loading AAPL_1.json
    # or showing a message for other symbols.
    ai_analysis_tab_content(st.session_state.selected_symbol_aianalysis)
else:
    st.info("Please enter a symbol in the sidebar and click 'Load AI Analysis'.")