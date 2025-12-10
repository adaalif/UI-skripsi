
import streamlit as st
import pandas as pd
from app.core.keyphrase_extraction import extract_keyphrases

def render_manual_tab(tokenizer, model, device):
    """
    Renders the manual input tab for keyphrase extraction.
    """
    st.header("✍️ Manual Document Input")
    st.markdown("Enter document title and content manually to extract keyphrases.")
    
    with st.form("manual_input_form"):
        title_input = st.text_input(
            "📌 Document Title",
            placeholder="e.g., Machine Learning in Healthcare",
            help="Enter the title of your document. If left blank, it will be inferred from the content."
        )
        
        text_input = st.text_area(
            "📄 Document Content",
            placeholder="Enter the full text content of your document here...",
            height=250,
            help="Enter the complete text content of your document (e.g., the abstract)."
        )
        
        top_k_manual = st.number_input(
            "Number of keyphrases (Top-K)",
            min_value=5,
            max_value=50,
            value=15,
            step=1,
            key="manual_top_k"
        )
        
        submit_manual = st.form_submit_button("🚀 Extract Keyphrases", type="primary", use_container_width=True)
    
    if submit_manual:
        if not text_input.strip():
            st.error("❌ Please enter document content.")
        else:
            with st.spinner("⏳ Running MuSe-Rank algorithm..."):
                try:
                    # Use title_input if provided, otherwise it will be inferred by extract_keyphrases
                    title_to_use = title_input.strip() if title_input.strip() else None
                    
                    keyphrase_results = extract_keyphrases(
                        text_input,
                        title_to_use,
                        tokenizer,
                        model,
                        device,
                        top_k=top_k_manual
                    )
                    
                    if not keyphrase_results:
                        st.warning("Tidak ditemukan frasa kunci yang valid. The text might be too short.")
                    else:
                        st.success(f"✅ Successfully extracted {len(keyphrase_results)} keyphrases!")
                        
                        df_results = pd.DataFrame(keyphrase_results, columns=["Keyphrase", "Score"])
                        df_results.index += 1
                        
                        min_score = df_results["Score"].min()
                        max_score = df_results["Score"].max()
                        df_results["Normalized Score"] = (df_results["Score"] - min_score) / (max_score - min_score) if max_score > min_score else 1
                        
                        st.dataframe(
                            df_results,
                            column_config={
                                "Keyphrase": st.column_config.TextColumn("Frasa Kunci", width="large"),
                                "Score": st.column_config.ProgressColumn(
                                    "Skor Relevansi",
                                    format="%.4f",
                                    min_value=0,
                                    max_value=1,
                                ),
                            },
                            use_container_width=True
                        )
                        
                        keyphrases_text = ", ".join(df_results["Keyphrase"].tolist())
                        st.download_button(
                            label="📥 Download Keyphrases as Text",
                            data=keyphrases_text,
                            file_name="manual_keyphrases.txt",
                            mime="text/plain"
                        )
                        
                except Exception as e:
                    st.error(f"❌ An error occurred during extraction: {e}")

