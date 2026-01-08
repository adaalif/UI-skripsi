import streamlit as st
import pandas as pd
from app.models.keyphrase_extractor import KeyphraseExtractor

class ManualInputTab:
    """
    Manages the UI and logic for the manual text input tab, following UC-01.
    """
    def __init__(self, extractor: KeyphraseExtractor):
        """
        Initializes the tab with a KeyphraseExtractor instance.
        """
        self.extractor = extractor

    def render(self):
        """
        Renders the Streamlit components for the manual input form.
        This corresponds to the main scenario where the user interacts with the UI.
        """
        st.header("UC-01: Ekstraksi Frasa Kunci Manual")
        st.markdown("Proses dokumen tunggal (judul dan abstrak) untuk mendapatkan daftar frasa kunci Top-K.")
        
        # Skenario Utama: Aktor mengisi form
        with st.form("manual_input_form"):
            title_input = st.text_input(
                "Judul Dokumen",
                placeholder="e.g., Machine Learning in Healthcare",
                help="Masukkan judul dokumen Anda. Jika kosong, akan diinferensi dari konten."
            )
            
            text_input = st.text_area(
                "Konten Dokumen (Abstrak)",
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
                key="manual_top_k",
                help="Atur jumlah frasa kunci teratas yang ingin ditampilkan."
            )
            
            # Skenario Utama: Aktor menekan tombol submit
            submit_manual = st.form_submit_button("Ekstrak Frasa Kunci", type="primary", use_container_width=True)
        
        if submit_manual:
            self._handle_submission(title_input, text_input, top_k_manual)

    def _handle_submission(self, title, content, top_k):
        """
        Handles the logic after the form is submitted, covering both successful and failed scenarios.
        """
        # Skenario Alternatif 1: Validasi Masukan Gagal
        if not content.strip():
            st.error("Mohon masukkan konten dokumen.")
            return # Proses berhenti

        # Skenario Utama: Sistem menjalankan algoritma
        with st.spinner("Menjalankan algoritma MuSe-Rank..."):
            try:
                # Menyiapkan data untuk ekstraksi
                title_to_use = title.strip() if title.strip() else None
                
                # Memanggil model untuk ekstraksi
                keyphrase_results = self.extractor.extract(
                    document_text=content,
                    title=title_to_use,
                    top_k=top_k
                )
                
                # Menampilkan hasil
                self._display_results(keyphrase_results)
                
            # Skenario Alternatif 3: Kesalahan Sistem
            except Exception as e:
                st.error(f"Terjadi kesalahan sistem saat pemrosesan: {e}")

    def _display_results(self, results):
        """
        Displays the extraction results or a warning if no keyphrases are found.
        """
        # Skenario Alternatif 2: Tidak Ditemukan Frasa Kunci
        if not results:
            st.warning("Tidak ditemukan frasa kunci yang valid. Teks mungkin terlalu pendek atau hanya berisi stopwords.")
            return

        # Skenario Utama: Sistem menampilkan hasil
        st.success(f"Berhasil mengekstrak {len(results)} frasa kunci!")
        
        df_results = pd.DataFrame(results)
        df_results.index += 1 # Membuat index mulai dari 1
        
        st.markdown("### Hasil Ekstraksi")
        st.info("Anda dapat mengurutkan hasil dengan mengklik pada nama kolom (misal: 'Skor Akhir').")
        
        # Menampilkan tabel hasil
        st.dataframe(df_results, use_container_width=True)
        
        # Menyiapkan data untuk diunduh
        keyphrases_text = "\n".join(df_results["Keyphrase"].tolist())
        
        # Skenario Utama: Sistem menampilkan tombol unduh
        st.download_button(
            label="Download Hasil (.txt)",
            data=keyphrases_text,
            file_name="hasil_ekstraksi_frasa_kunci.txt",
            mime="text/plain"
        )
