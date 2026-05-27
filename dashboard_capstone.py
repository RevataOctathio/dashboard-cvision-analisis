import streamlit as st
import pandas as pd
import matplotlib.subplots as plt
import seaborn as sns

# Set up Streamlit page configuration
st.set_page_config(
    page_title="CVision Analytics Dashboard", 
    layout="wide", 
    page_icon="📊"
)
sns.set_theme(style="whitegrid")

# DATA LOADING FUNCTION (CACHE)
@st.cache_data
def load_dashboard_data():
    try:
        # Memuat dataset final yang sudah melalui tahap Oversampling dan Lemmatization
        df_main = pd.read_csv('df_final.csv')
        q1 = pd.read_csv('streamlit_data/q1_kategori_counts.csv')
        q2 = pd.read_csv('streamlit_data/q2_top_10_jobs.csv')
        q3 = pd.read_csv('streamlit_data/q3_top_skills_matrix.csv')
        q4 = pd.read_csv('streamlit_data/q4_hybrid_trend.csv')
        q5 = pd.read_csv('streamlit_data/q5_word_count_stats.csv')
        
        # Fallback perhitungan word_count jika belum tersedia di CSV
        if 'word_count' not in df_main.columns:
            df_main['word_count'] = df_main['desc_clean'].astype(str).apply(lambda x: len(x.split()))
            
        return q1, q2, q3, q4, q5, df_main
    except FileNotFoundError:
        return None, None, None, None, None, None

q1_data, q2_data, q3_data, q4_data, q5_data, df_main = load_dashboard_data()

# HEADER
st.title("Dashboard Analisis CVision")
st.markdown("##### Analisis Eksploratif Kualifikasi Pekerjaan untuk Optimalisasi Sistem Pencocokan Resume Berbasis AI")
st.divider()

# SIDEBAR
if q1_data is None or df_main is None:
    st.error("Berkas data tidak ditemukan. Pastikan df_final.csv dan folder streamlit_data tersedia.")
else:
    st.sidebar.title("CVision")
    
    selected_q = st.sidebar.selectbox(
        "Pilih Analisis:",
        options=[
            "Pertanyaan 1: Distribusi Volume Lowongan",
            "Pertanyaan 2: Top Posisi Pekerjaan Populer",
            "Pertanyaan 3: Karakteristik DNA Kosakata",
            "Pertanyaan 4: Analisis Pekerjaan Hybrid",
            "Pertanyaan 5: Kepadatan Informasi Teks"
        ]
    )
    st.sidebar.markdown("""
        <div style='position: fixed; bottom: 20px; text-align: left; color: #64748B; font-size: 13px; line-height: 1.5;'>
            <strong>Capstone Project 'CVision'</strong><br>
            Data Science<br>
            CC26 - PSU100
        </div>
    """, unsafe_allow_html=True)
    
    # PERTANYAAN 1 (Bar Chart Vertikal & Anotasi Persentase)
    if selected_q.startswith("Pertanyaan 1"):
        st.markdown("### 📊 Visualisasi Grafik")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        
        sns.barplot(x='Kategori Industri', y='Jumlah Lowongan', data=q1_data, hue='Kategori Industri', palette='muted', legend=False, ax=ax1)
        
        plt.title('Proporsi Volume Data per Sektor Kategori', fontsize=11, weight='bold')
        
        total_lowongan = q1_data['Jumlah Lowongan'].sum()
        
        for p in ax1.patches:
            height = p.get_height()
            percentage = (height / total_lowongan) * 100
            ax1.annotate(f"{int(height):,} ({percentage:.2f}%)", 
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', weight='bold', color='white', fontsize=9)
        st.pyplot(fig1)
        
        # PERBAIKAN NARASI: Penyesuaian dengan teknik Oversampling di Notebook
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nGrafik menunjukkan 5 pilar industri dengan proporsi yang rata sempurna (masing-masing 10.000 data atau 20.00%). Kondisi ekuilibrium ini merupakan hasil dari penerapan teknik **Oversampling** pada tahap *Data Preparation* untuk mengatasi ketimpangan jumlah data mentah (*Imbalanced Data*).")
        st.success("**Conclusion:**\n\nKeberhasilan teknik *Oversampling* dalam menyeimbangkan distribusi kelas memastikan dataset terbebas mutlak dari risiko *Majority Class Bias*. Arsitektur Machine Learning kini dapat mengekstraksi fitur dan memberikan rekomendasi karir secara adil, objektif, dan tanpa diskriminasi terhadap profesi minoritas.")

    # PERTANYAAN 2 (Bar Chart Horizontal & Slider)
    elif selected_q.startswith("Pertanyaan 2"):
        top_n = st.slider("Atur Batas Tampilan Peringkat Pekerjaan:", min_value=3, max_value=10, value=10)
        filtered_q2 = q2_data.head(top_n)
        
        st.markdown("### 📊 Visualisasi Grafik")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        
        sns.barplot(x='Jumlah Lowongan', y='Posisi Pekerjaan (Job Title)', data=filtered_q2, hue='Posisi Pekerjaan (Job Title)', palette='muted', legend=False, ax=ax2)
        
        plt.title(f'Peringkat {top_n} Besar Jabatan Lowongan Kerja Terpopuler', fontsize=11, weight='bold')
        st.pyplot(fig2)
        
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nVisualisasi grafik batang horizontal memperlihatkan secara jelas dominasi pasar tenaga kerja oleh rumpun komersial dan lini depan. Posisi 'sales executive' mendominasi grafik dengan volume rekrutmen tertinggi (777 lowongan).")
        st.success("**Conclusion:**\n\nTingginya volume rekrutmen pada posisi lini depan komersial (dipimpin oleh Sales Executive dengan 777 lowongan) dan layanan kesehatan merepresentasikan fokus serapan pasar tenaga kerja saat ini. Pengembangan UI/UX pada dashboard aplikasi perlu memprioritaskan penyajian 'High Demand Jobs' pada halaman utama untuk mengarahkan pengguna ke sektor dengan probabilitas rekrutmen tertinggi.")

    # PERTANYAAN 3 (Heatmap & Anotasi Bobot Keterampilan)
    elif selected_q.startswith("Pertanyaan 3"):
        q3_heatmap = q3_data.set_index('Rank')
        
        st.markdown("### 📊 Visualisasi Grafik")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        mock_weights = pd.DataFrame(index=q3_heatmap.index, columns=q3_heatmap.columns)
        for col in q3_heatmap.columns:
            mock_weights[col] = range(10, 0, -1)
            
        sns.heatmap(mock_weights, annot=q3_heatmap, fmt="", cmap='YlGnBu', cbar=False, ax=ax3, annot_kws={"size": 9})
        plt.title('Matriks Pemisahan Semantik Kerapatan Unigram Utama per Industri', fontsize=11, weight='bold')
        st.pyplot(fig3)
        
        # PERBAIKAN NARASI: Penyesuaian dengan teknik Lemmatization
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nSetelah teks melalui pembersihan tingkat lanjut (*Lemmatization* dan Regex), heatmap membuktikan keberadaan \"DNA Kosakata\" yang sangat spesifik. Kata universal terpetakan beririsan, sementara istilah sektoral (seperti 'medical' di Healthcare dan 'engineering' di Engineering) terkunci pekat. Pembersihan *noise* ini menghasilkan pemisah kosakata yang tegas dan murni.")
        st.success("**Conclusion:**\n\nKeberadaan 'DNA Kosakata' yang telah terbebas dari residu kata hubung ini membuktikan urgensi penyaringan semantik. Tim AI Engineer siap mengeksekusi algoritma TF-IDF untuk mengubah dataset bersih ini menjadi matriks vektor, memastikan akurasi pencocokan resume menggunakan Cosine Similarity berjalan presisi dan efisien.")

    # PERTANYAAN 4 (Bar Chart Vertikal)
    elif selected_q.startswith("Pertanyaan 4"):
        st.markdown("### 📊 Visualisasi Grafik")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        
        sns.barplot(x='Kategori Non-IT', y='Persentase dari Total Kategori (%)', data=q4_data, hue='Kategori Non-IT', palette='muted', legend=False, ax=ax4)
        
        plt.title('Proporsi Lowongan Kerja Non-IT yang Menuntut Kompetensi Digital', fontsize=11, weight='bold')
        for p in ax4.patches:
            ax4.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', weight='bold', color='white', fontsize=10)
        st.pyplot(fig4)
        
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nGrafik proporsi pekerjaan secara gamblang memvalidasi masifnya penetrasi kompetensi digital di sektor non-IT. Terlihat jelas bahwa baris Engineering (67.10%) and Business & Admin (65.97%) mendominasi kebutuhan literasi teknologi, membuktikan bahwa algoritma AI kelak sangat relevan untuk menyokong rekomendasi peran hybrid kepada pelamar non-teknis.")
        st.success("**Conclusion:**\n\nDominasi kebutuhan literasi digital pada sektor non-IT memvalidasi tren ekspansi pekerjaan hybrid. Algoritma rekomendasi harus dikonfigurasi agar secara proaktif memfasilitasi Career Pivot, mengarahkan kandidat berlatar belakang non-teknis yang memiliki kompetensi digital menuju peluang lintas industri tersebut.")

    # PERTANYAAN 5 (Kombinasi Box Plot, Bar Chart Interaktif, dan Tabel Angka Deskriptif)
    elif selected_q.startswith("Pertanyaan 5"):
        st.markdown("### 📊 Visualisasi Grafik (Box Plot)")
        
        if df_main is None:
            st.error("⚠️ Dataset utama tidak ditemukan. Pastikan file df_final.csv berada di direktori yang benar.")
        else:
            fig_box, ax_box = plt.subplots(figsize=(12, 6))
            
            sns.boxplot(
                x='word_count',
                y='main_category',
                data=df_main,
                hue='main_category',
                fliersize=3,
                ax=ax_box
            )
            
            if ax_box.get_legend() is not None:
                ax_box.get_legend().remove()
                
            plt.title('Analisis Distribusi Panjang Kata (Word Count) Deskripsi Lowongan per Industri', fontsize=14, pad=15, weight='bold')
            plt.xlabel('Jumlah Kata dalam Teks Deskripsi (Word Count)', fontsize=12)
            plt.ylabel('Kategori Industri', fontsize=12)
            
            plt.xlim(-50, df_main['word_count'].max() + 100)
            plt.tight_layout()
            
            st.pyplot(fig_box)
        
        st.markdown("##### Tabel Angka Deskriptif Parameter Kalibrasi AI")
        st.dataframe(q5_data.set_index('Kategori Industri'), use_container_width=True)
        
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nVisualisasi *boxplot* dan tabel mendemonstrasikan hierarki kepadatan informasi (*information density*) secara komprehensif pada kolom deskripsi yang telah dibersihkan. Kategori 'Information Technology' secara konsisten menduduki peringkat teratas, baik pada sebaran nilai tengah (median 202 kata) maupun batas pencilan ekstrem (outliers).")
        st.success("**Conclusion:**\n\nAnalisis kepadatan teks menunjukkan variansi ekstrem antar-industri. Untuk menjaga efisiensi RAM dan performa server aplikasi saat sistem memproses ribuan data, parameter batas token teks (*max_features*) pada model TF-IDF harus dikalibrasi di kisaran angka median tertinggi (yakni sektor IT: 202 kata). Penetapan parameter berbasis bukti kuantitatif ini mencegah risiko komputasi *overfitting*.")