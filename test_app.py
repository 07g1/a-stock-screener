import streamlit as st

st.set_page_config(page_title="Test", page_icon="✅")
st.title("✅ 部署成功！")
st.write("如果你能看到这个页面，说明 Streamlit Cloud 部署没问题。")
st.write(f"Streamlit version: {st.__version__}")

import sys
st.write(f"Python version: {sys.version}")