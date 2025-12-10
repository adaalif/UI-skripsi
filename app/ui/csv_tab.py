
import streamlit as st
import pandas as pd
from app.core.keyphrase_extraction import extract_keyphrases

def render_csv_tab(tokenizer, model, device):
    """
    Renders the CSV upload tab for batch keyphrase extraction.
    """
    st.header("📁 Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV file should contain columns for title and text content."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Found {len(df)} rows.")
            
            st.subheader("📊 CSV Preview and Column Selection")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("Please select the columns containing the document titles and text content.")
            col1, col2 = st.columns(2)
            with col1:
                title_col = st.selectbox("Select Title Column", df.columns, index=0)
            with col2:
                text_col = st.selectbox("Select Text/Abstract Column", df.columns, index=1 if len(df.columns) > 1 else 0)

            st.header("⚙️ Processing Options")
            top_k = st.number_input("Number of keyphrases per document", min_value=5, max_value=50, value=15, step=5, key="csv_top_k")
            
            if st.button("🚀 Extract Keyphrases from CSV", type="primary", use_container_width=True):
                rows_to_process = df
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in rows_to_process.iterrows():
                    status_text.text(f"Processing row {idx + 1}/{len(rows_to_process)}...")
                    progress_bar.progress((idx + 1) / len(rows_to_process))
                    
                    title = str(row[title_col]) if pd.notna(row[title_col]) else ""
                    text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                    
                    if not title or not text:
                        keyphrase_results = []
                    else:
                        keyphrase_results = extract_keyphrases(text, title, tokenizer, model, device, top_k=top_k)
                    
                    results.append({
                        'title': title,
                        'keyphrases': ", ".join([p for p, s in keyphrase_results])
                    })

                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Processed {len(results)} documents!")
                
                # Create a new DataFrame for the results
                results_df = pd.DataFrame(results)
                
                st.header("📋 Extraction Results")
                st.dataframe(results_df, use_container_width=True)
                
                csv_output = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_output,
                    file_name="keyphrase_extraction_results.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"❌ Error processing CSV file: {e}")
            st.exception(e)
    else:
        st.info("👆 Upload a CSV file to get started.")
