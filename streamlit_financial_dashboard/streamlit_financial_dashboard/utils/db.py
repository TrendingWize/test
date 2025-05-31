import streamlit as st
from neo4j import GraphDatabase

@st.cache_resource
def get_neo4j_driver():
    try:
        uri = st.secrets["neo4j"]["NEO4J_URI"]
        user = st.secrets["neo4j"]["NEO4J_USER"]
        password = st.secrets["neo4j"]["NEO4J_PASSWORD"]
    except KeyError as e:
        raise ValueError(f"Missing Neo4j secret: {e}")

    return GraphDatabase.driver(uri, auth=(user, password))
