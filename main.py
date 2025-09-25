import streamlit as st

from watermark_page import show_watermark_page
from page_2 import show_page_2

# Define your pages as StreamlitPage objects
page1 = st.Page(show_watermark_page, title="Watermark")
page2 = st.Page(show_page_2, title="Page 2")

# Create the navigation menu
pg = st.navigation([page1, page2])

pg.run()
