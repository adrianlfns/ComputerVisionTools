import streamlit as st

from watermark_page import show_watermark_page
from pdf_sign_page import show_pdf_sign_page

# Define your pages as StreamlitPage objects
page1 = st.Page(show_watermark_page, title="Watermark", icon="✒")
page2 = st.Page(show_pdf_sign_page, title="PDF Sign", icon="📝")

# Create the navigation menu
pg = st.navigation([page1, page2])

pg.run()
