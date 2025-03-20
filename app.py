import streamlit as st
import os
from main import research, PDF_FILES, TEXT_FILES

# Streamlit App
def main():
    st.title("Multi-Source Bot")
    st.write("Enter a research query below to get results from various sources.")

    # Check if files exist at startup and display warnings
    for pdf in PDF_FILES:
        if not os.path.exists(pdf):
            st.warning(f"Warning: PDF file '{pdf}' not found. Update PDF_FILES in main.py.")

    for txt in TEXT_FILES:
        if not os.path.exists(txt):
            st.warning(f"Warning: Text file '{txt}' not found. Update TEXT_FILES in main.py.")

    # Input query
    query = st.text_input("Research Query", placeholder="e.g., What is LLM?")

    if st.button("Research"):
        if query:
            with st.spinner("Researching... This may take a moment."):
                result = research(query)
            st.subheader("Research Results")
            st.write(result)
        else:
            st.error("Please enter a query to research.")

if __name__ == "__main__":
    main()
