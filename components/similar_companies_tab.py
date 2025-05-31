import streamlit as st
import pandas as pd
import numpy as np # For type hints if used directly
from utils import (
    get_neo4j_driver, 
    get_nearest_aggregate_similarities, 
    fetch_financial_details_for_companies,
    format_value # For displaying financial values nicely
)
from typing import List, Tuple, Dict # For type hints

def similar_companies_tab_content():
    st.title("🔗 Find Similar Companies")
    st.markdown("Discover companies with similar financial statement characteristics based on embeddings.")

    # --- Inputs ---
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        target_symbol_input = st.text_input("Enter Target Company Symbol (e.g., NVDA, AAPL)", 
                                            value=st.session_state.get("similar_target_sym", "NVDA"), 
                                            key="similar_target_sym_input").upper()
    with col2:
        embedding_family_options = {
            "Cash Flow (cf_vec_)": "cf_vec_",
            "Income Statement (is_vec_)": "is_vec_",
            "Balance Sheet (bs_vec_)": "bs_vec_",
            # "All Statements (all_vec_)": "all_vec_" # Add if you have these
        }
        selected_family_display = st.selectbox("Select Similarity Type (Embedding Family)", 
                                               options=list(embedding_family_options.keys()),
                                               index=0, # Default to Cash Flow
                                               key="similar_embedding_family_select")
        embedding_family_value = embedding_family_options[selected_family_display]
    
    # Year range for similarity aggregation (can be made configurable too)
    start_similarity_year = 2020 # More recent data might be better
    end_similarity_year = 2025   # Up to the latest full year of vector data you have
    num_similar_companies = st.slider("Number of Similar Companies to Display", min_value=5, max_value=25, value=10, step=1, key="similar_k_slider")


    if 'similar_companies_data' not in st.session_state:
        st.session_state.similar_companies_data = None
    if 'similar_companies_details' not in st.session_state:
        st.session_state.similar_companies_details = None
    if 'last_target_sym_processed' not in st.session_state:
        st.session_state.last_target_sym_processed = None
    if 'last_embedding_family_processed' not in st.session_state:
        st.session_state.last_embedding_family_processed = None


    search_button_col, _ = st.columns([1,3]) # To make button not too wide
    with search_button_col:
        if st.button("🚀 Find Similar Companies", key="find_similar_btn", type="primary", use_container_width=True):
            if not target_symbol_input:
                st.warning("Please enter a target company symbol.")
                return # Use st.stop() if preferred
            
            st.session_state.similar_target_sym = target_symbol_input # Save for next time
            st.session_state.similar_companies_data = None # Reset previous results
            st.session_state.similar_companies_details = None

            neo_driver = get_neo4j_driver()
            if not neo_driver:
                st.error("Database connection failed.")
                return

            with st.spinner(f"Finding companies similar to {target_symbol_input} using {selected_family_display}..."):
                similar_companies = get_nearest_aggregate_similarities(
                    _driver=neo_driver,
                    target_sym=target_symbol_input,
                    embedding_family=embedding_family_value,
                    start_year=start_similarity_year,
                    end_year=end_similarity_year,
                    k=num_similar_companies
                )
                st.session_state.similar_companies_data = similar_companies
                st.session_state.last_target_sym_processed = target_symbol_input
                st.session_state.last_embedding_family_processed = selected_family_display

                if similar_companies:
                    symbols_to_fetch = [sym for sym, score in similar_companies]
                    financial_details = fetch_financial_details_for_companies(neo_driver, symbols_to_fetch)
                    st.session_state.similar_companies_details = financial_details
                    st.write("Symbols returned by Python script for financial details lookup:", symbols_to_fetch)
                else:
                    st.info(f"No similar companies found for {target_symbol_input} with the selected criteria.")
            st.rerun() # Rerun to display results below

    st.markdown("---")

    # --- Display Results ---
    if st.session_state.get("last_target_sym_processed"):
        st.subheader(f"Top {num_similar_companies} Companies Similar to {st.session_state.last_target_sym_processed} (using {st.session_state.last_embedding_family_processed})")

    if st.session_state.similar_companies_data:
        similar_companies_list = st.session_state.similar_companies_data
        financial_details_dict = st.session_state.similar_companies_details or {}

        if not similar_companies_list:
            st.info(f"No similar companies found for {st.session_state.last_target_sym_processed} or data processing error.")
        
        for i, (sym, score) in enumerate(similar_companies_list):
            details = financial_details_dict.get(sym, {})
            company_name = details.get("companyName", sym) # Fallback to symbol if name not found
            sector = details.get("sector", "N/A")
            industry = details.get("industry", "N/A")

            with st.container(): # Use a container for each company for better visual separation
                st.markdown(f"**{i+1}. {company_name} ({sym})** - Similarity Score: `{score:.4f}`")
                st.caption(f"Sector: {sector} | Industry: {industry}")

                cols_metrics = st.columns(3) # For IS, BS, CF metrics side-by-side
                with cols_metrics[0]:
                    st.markdown("###### Income Statement")
                    st.metric("Revenue", format_value(details.get("revenue")))
                    st.metric("Net Income", format_value(details.get("netIncome")))
                    st.metric("Operating Income", format_value(details.get("operatingIncome")))
                    st.metric("Gross Profit", format_value(details.get("grossProfit")))
                
                with cols_metrics[1]:
                    st.markdown("###### Balance Sheet")
                    st.metric("Total Assets", format_value(details.get("totalAssets")))
                    st.metric("Total Liabilities", format_value(details.get("totalLiabilities")))
                    st.metric("Equity", format_value(details.get("totalStockholdersEquity")))
                    st.metric("Cash", format_value(details.get("cashAndCashEquivalents")))

                with cols_metrics[2]:
                    st.markdown("###### Cash Flow")
                    st.metric("Operating CF", format_value(details.get("operatingCashFlow")))
                    st.metric("Free CF", format_value(details.get("freeCashFlow")))
                    st.metric("Net Change in Cash", format_value(details.get("netChangeInCash")))
                    st.metric("CapEx", format_value(details.get("capitalExpenditure")))
                st.markdown("---") # Separator between companies
    elif st.session_state.get("last_target_sym_processed"): # If processed but no results
        st.info(f"No similar companies found for {st.session_state.last_target_sym_processed} with the selected criteria after processing.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    # Mocking for standalone run (important!)
    class MockNeo4jDriver:
        def session(self, database=None): return self
        def __enter__(self): return self
        def __exit__(self, type, value, traceback): pass
        def run(self, query, **kwargs): 
            print(f"Mock RUN: {query[:100]}... with {kwargs}") # Print query for debugging
            if "c.cf_vec_2022 IS NOT NULL" in query and kwargs.get('sym') == 'NVDA': # Example condition
                 return [{"sym": "AMD", "vec": np.random.rand(10).tolist()}, {"sym": "INTC", "vec": np.random.rand(10).tolist()}]
            if "UNWIND $symbols AS sym_param" in query:
                mock_details = []
                for s_param in kwargs.get("symbols", []):
                    mock_details.append({
                        "symbol": s_param, "companyName": f"{s_param} Inc.", "sector": "Tech", "industry": "Semiconductors",
                        "revenue": 1000, "netIncome": 100, "operatingIncome": 200, "grossProfit": 500,
                        "totalAssets": 2000, "totalLiabilities": 800, "totalStockholdersEquity": 1200, "cashAndCashEquivalents": 300,
                        "operatingCashFlow": 250, "freeCashFlow": 150, "netChangeInCash": 50, "capitalExpenditure": -100
                    })
                return mock_details
            return [] # Default empty result
        def verify_connectivity(self): pass

    def get_neo4j_driver(): 
        print("Using Mock Neo4j Driver for similar_companies_tab standalone run.")
        return MockNeo4jDriver()
    
    def format_value(value, is_percent=False): # Basic mock
        if pd.isna(value) or value is None: return "N/A"
        return f"${value/1e6:.2f}M" if abs(value) >= 1e6 else f"${value:,.0f}"

    # You might need to provide mock implementations for other utils functions if they are called and not robust to missing driver
    
    similar_companies_tab_content()