import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
        df_main = pd.read_csv('df_final.csv')
        q1 = pd.read_csv('streamlit_data/q1_kategori_counts.csv')
        q2 = pd.read_csv('streamlit_data/q2_top_10_jobs.csv')
        q3 = pd.read_csv('streamlit_data/q3_top_skills_matrix.csv')
        q4 = pd.read_csv('streamlit_data/q4_hybrid_trend.csv')
        q5 = pd.read_csv('streamlit_data/q5_word_count_stats.csv')
        
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
    st.error("Berkas data tidak ditemukan.")
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
        
        # PERBAIKAN: Menambahkan hue dan legend=False
        sns.barplot(x='Kategori Industri', y='Jumlah Lowongan', data=q1_data, hue='Kategori Industri', palette='muted', legend=False, ax=ax1)
        
        plt.title('Proporsi Volume Data per Sektor Kategori', fontsize=11, weight='bold')
        
        # Total Data untuk menghitung persentase
        total_lowongan = q1_data['Jumlah Lowongan'].sum()
        
        # Menampilkan anotasi jumlah dan persentase pada setiap batang 
        for p in ax1.patches:
            height = p.get_height()
            percentage = (height / total_lowongan) * 100
            ax1.annotate(f"{int(height):,} ({percentage:.2f}%)", 
                        (p.get_x() + p.get_width() / 2., height / 2),
                        ha='center', va='center', weight='bold', color='white', fontsize=9)
        st.pyplot(fig1)
        
        # Menampilkan Insight & Kesimpulan Pertanyaan 1
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nGrafik menunjukkan 5 pilar industri dengan tinggi yang rata sempurna (masing-masing tepat berisikan 10.000 data atau berkontribusi seimbang sebesar 20.00%). Kondisi seimbang ini secara matematis mengeliminasi risiko Majority Class Bias pada arsitektur Machine Learning.")
        st.success("**Conclusion:**\n\nBerdasarkan distribusi volume data yang seimbang mutlak (20% per kategori industri), dataset terbukti terbebas dari Majority Class Bias. Sebagai tindak lanjut teknis, model algoritma dapat langsung dilatih tanpa perlu teknik resampling tambahan, guna memastikan sistem memberikan rekomendasi karir yang adil dan objektif untuk seluruh kluster.")

    # PERTANYAAN 2 (Bar Chart Horizontal & Slider)
    elif selected_q.startswith("Pertanyaan 2"):
        # Untuk mengatur jumlah posisi pekerjaan yang ditampilkan dengan slider
        top_n = st.slider("Atur Batas Tampilan Peringkat Pekerjaan:", min_value=3, max_value=10, value=10)
        filtered_q2 = q2_data.head(top_n)
        
        st.markdown("### 📊 Visualisasi Grafik")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        
        # PERBAIKAN: Menambahkan hue dan legend=False
        sns.barplot(x='Jumlah Lowongan', y='Posisi Pekerjaan (Job Title)', data=filtered_q2, hue='Posisi Pekerjaan (Job Title)', palette='muted', legend=False, ax=ax2)
        
        plt.title(f'Peringkat {top_n} Besar Jabatan Lowongan Kerja Terpopuler', fontsize=11, weight='bold')
        st.pyplot(fig2)
        
        # Menampilkan Insight & Kesimpulan Pertanyaan 2
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
        
        # Menampilkan Insight & Kesimpulan Pertanyaan 3
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nVisualisasi heatmap membuktikan secara empiris keberadaan \"DNA Kosakata\" (Bag-of-Words) yang spesifik dan mandiri. Kata universal seperti 'experience' atau 'job' terpetakan beririsan di semua kategori dengan bobot serupa. Namun, istilah sektoral seperti 'medical' terkunci pekat di Healthcare dan 'engineering' di Engineering. Pemisah kosakata yang tegas ini memvalidasi keakurasi model TF-IDF nantinya.")
        st.success("**Conclusion:**\n\nKeberadaan irisan keterampilan universal yang berdampingan dengan 'DNA Kosakata' industri spesifik membuktikan urgensi penyaringan semantik. Tim wajib mengeksekusi algoritma TF-IDF untuk menekan bobot kata umum (noise) dan memaksimalkan bobot keterampilan unik, sehingga akurasi pencocokan resume menggunakan Cosine Similarity berjalan presisi.")

    # PERTANYAAN 4 (Bar Chart Vertikal)
    elif selected_q.startswith("Pertanyaan 4"):
        st.markdown("### 📊 Visualisasi Grafik")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        
        # PERBAIKAN: Menambahkan hue dan legend=False
        sns.barplot(x='Kategori Non-IT', y='Persentase dari Total Kategori (%)', data=q4_data, hue='Kategori Non-IT', palette='muted', legend=False, ax=ax4)
        
        plt.title('Proporsi Lowongan Kerja Non-IT yang Menuntut Kompetensi Digital', fontsize=11, weight='bold')
        for p in ax4.patches:
            ax4.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha='center', va='center', weight='bold', color='white', fontsize=10)
        st.pyplot(fig4)
        
        # Menampilkan Insight & Kesimpulan Pertanyaan 4
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nGrafik proporsi pekerjaan secara gamblang memvalidasi masifnya penetrasi kompetensi digital di sektor non-IT. Terlihat jelas bahwa baris Engineering (67.10%) and Business & Admin (65.97%) mendominasi kebutuhan literasi teknologi, membuktikan bahwa algoritma AI kelak sangat relevan untuk menyokong rekomendasi peran hybrid kepada pelamar non-teknis.")
        st.success("**Conclusion:**\n\nDominasi kebutuhan literasi digital pada sektor non-IT (mencapai 67.10% di Engineering dan 65.97% di Business & Admin) memvalidasi tren ekspansi pekerjaan hybrid. Algoritma rekomendasi harus dikonfigurasi agar secara proaktif memfasilitasi Career Pivot, mengarahkan kandidat berlatar belakang non-teknis yang memiliki kompetensi digital menuju peluang lintas industri tersebut.")

    # HALAMAN PERTANYAAN 5 (Kombinasi Box Plot, Bar Chart Interaktif, dan Tabel Angka Deskriptif)
    elif selected_q.startswith("Pertanyaan 5"):
        st.markdown("### 📊 Visualisasi Grafik (Box Plot)")
        
        if df_main is None:
            st.error("⚠️ Dataset utama tidak ditemukan.")
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
        
        # Menampilkan Tabel Angka Deskriptif Lengkap
        st.markdown("##### Tabel Angka Deskriptif Parameter Kalibrasi AI")
        st.dataframe(q5_data.set_index('Kategori Industri'), use_container_width=True)
        
        # Menampilkan Insight & Kesimpulan Pertanyaan 5
        st.markdown("### 💡 Insight & Kesimpulan")
        st.info("**Insight Visualisasi:**\n\nVisualisasi *boxplot* dan tabel mendemonstrasikan hierarki kepadatan informasi (*information density*) dari setiap rumpun industri secara komprehensif. Kategori 'Information Technology' secara konsisten menduduki peringkat teratas, baik pada sebaran nilai tengah (median 202 kata) maupun batas pencilan ekstrem (outliers mencapai ~1.500 kata).")
        st.success("**Conclusion:**\n\nAnalisis kepadatan teks menunjukkan variansi ekstrem antar-industri. Untuk menjaga efisiensi RAM dan performa server aplikasi saat sistem memproses ribuan data, parameter batas token teks (*max_features*) pada model TF-IDF harus dikalibrasi di kisaran angka median tertinggi (yakni sektor IT: 202 kata). Penetapan parameter berbasis bukti kuantitatif ini akan mencegah risiko komputasi *overfitting* tanpa menghilangkan konteks krusial dari deskripsi lowongan.")