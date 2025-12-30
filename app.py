# APP VERSION 1: Hello Streamlit 👋
# This whole cell is a complete Streamlit app.
# Copy EVERYTHING in this cell into a file named: app.py
# Then run in your terminal:
#     streamlit run app.py
#
# Tip: Streamlit reruns the script top-to-bottom whenever you change a widget.


import streamlit as st

# st.set_page_config changes the browser tab title + layout
st.set_page_config(page_title="My First Streamlit App", layout="centered")

# st.title makes a big title
st.title("Hello, Streamlit! 👋")

# st.write is a super-flexible printer (text, numbers, lists, even tables)
st.write("This is my first Streamlit app!")

# You can add emojis and markdown too
st.markdown("**Cool!** Now let's add buttons and inputs in the next version.")
