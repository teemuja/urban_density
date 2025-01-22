import streamlit as st

st.set_page_config(page_title="Research App", layout="wide", initial_sidebar_state='expanded')
st.markdown("""
<style>
button[title="View fullscreen"]{
        visibility: hidden;}
</style>
""", unsafe_allow_html=True)

#title
st.header("Title",divider='green')


#sidebar footer
st.markdown('---')
footer_title = '''
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/)
'''
st.markdown(footer_title)
disclamer = 'Data papers are constant work in progress and will be upgraded, changed & fixed while research go on.'
st.caption('Disclaimer: ' + disclamer)