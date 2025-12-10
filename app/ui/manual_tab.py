import streamlit as st
import pandas as pd
from app.core.keyphrase_extraction import KeyphraseExtractor

class ManualInputTab:
    """
    Manages the UI and logic for the manual text input tab.
    """
    def __init__(self, extractor: KeyphraseExtractor):
        """
        Initializes the tab with a KeyphraseExtractor instance.
        """
        self.extractor = extractor

    def render(self):
        """
        Renders the Streamlit components for the manual input form.
        """
        st.header("Input Teks")
        st.markdown("Masukkan judul dan konten dokumen secara manual untuk mengekstrak frasa kunci.")
        
        with st.form("manual_input_form"):
            title_input = st.text_input(
                "Judul Dokumen",
                placeholder="e.g., Machine Learning in Healthcare",
                help="Masukkan judul dokumen Anda. Jika kosong, akan diinferensi dari konten."
            )
            
            text_input = st.text_area(
                "Konten Dokumen",
                placeholder="Masukkan konten lengkap dokumen Anda di sini...",
                height=250,
                help="Masukkan teks lengkap dari dokumen Anda (misal: abstrak)."
            )
            
            top_k_manual = st.number_input(
                "Jumlah frasa kunci yang diinginkan (Top-K)",
                min_value=5,
                max_value=50,
                value=15,
                step=1,
                key="manual_top_k"
            )
            
            submit_manual = st.form_submit_button("Ekstrak Frasa Kunci", type="primary", use_container_width=True)
        
        if submit_manual:
            self._handle_submission(title_input, text_input, top_k_manual)

    def _handle_submission(self, title_input, text_input, top_k):
        """
        Private method to handle the logic after the form is submitted.
        """
        if not text_input.strip():
            st.error("Mohon masukkan konten dokumen.")
        else:
            with st.spinner("Menjalankan algoritma MuSe-Rank..."):
                try:
                    title_to_use = title_input.strip() if title_input.strip() else None
                    
                    keyphrase_results = self.extractor.extract(
                        document_text=text_input,
                        title=title_to_use,
                        top_k=top_k
                    )
                    
                    self._display_results(keyphrase_results)
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat ekstraksi: {e}")

    def _display_results(self, keyphrase_results):
        """
        Private method to display the extraction results with detailed scores.
        """
        if not keyphrase_results:
            st.warning("Tidak ditemukan frasa kunci yang valid. Teks mungkin terlalu pendek.")
        else:
            st.success(f"Berhasil mengekstrak {len(keyphrase_results)} frasa kunci!")
            
            df_results = pd.DataFrame(keyphrase_results)
            df_results.index += 1
            
            st.markdown("### Hasil Ekstraksi")
            st.info("Anda dapat mengurutkan hasil dengan mengklik pada nama kolom (misal: klik pada 'Skor Akhir').")
            
            st.dataframe(df_results, use_container_width=True)
            
            keyphrases_text = ", ".join(df_results["Keyphrase"].tolist())
            st.download_button(
                label="Download Frasa Kunci (.txt)",
                data=keyphrases_text,
                file_name="manual_keyphrases.txt",
                mime="text/plain"
            )
