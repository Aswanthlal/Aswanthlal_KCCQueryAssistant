import asyncio
import nest_asyncio
import streamlit as st
import chromadb
import torch
import types
from chromadb import PersistentClient
from rag_pipeline import search_kcc

# ✅ Apply Streamlit page config as FIRST Streamlit command
st.set_page_config(page_title='KCC Query Assistant', layout='wide')

# ✅ Ensure asyncio loop is set
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ✅ Patch nested event loops
nest_asyncio.apply()

# ✅ Monkey-patch torch.classes for Streamlit compatibility
if isinstance(torch.classes, types.ModuleType):
    if not hasattr(torch.classes, "__path__"):
        torch.classes.__path__ = []

# ✅ Connect to ChromaDB
client = PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="kcc_data_1_8th", embedding_function=embedding_function)


# ✅ Debug document count in sidebar
st.sidebar.write("📊 ChromaDB contains", collection.count(), "documents.")

# ✅ Streamlit UI
st.title('KCC Query Assistant')
st.markdown('"Ask anything about agriculture based on Kisan Call Center advice.')

user_query = st.text_input('ENTER YOUR QUERY')

if st.button('Get Advice') and user_query.strip():
    with st.spinner('searching local knowledge base'):
        response = search_kcc(user_query)
        st.markdown('### Response')
        st.success(response)
