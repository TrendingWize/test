
import os
from neo4j import GraphDatabase
import streamlit as st
@st.cache_resource


@st.cache_resource
def get_neo4j_driver():
    uri = st.secrets["neo4j"]["NEO4J_URI"]
    user = st.secrets["neo4j"]["NEO4J_USER"]
    password = st.secrets["neo4j"]["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password))

