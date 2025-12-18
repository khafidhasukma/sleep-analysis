"""
Aplikasi Prediksi Gangguan Tidur dengan Random Forest Classifier
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi
st.set_page_config(
    page_title="Cek Gangguan Tidur",
    page_icon="💤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling minimal
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class SleepDisorderPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = []
        self.accuracy = 0
        self.class_labels = []
        
    def load_and_prepare_data(self, filepath):
        df = pd.read_csv(filepath)
        
        # Jangan drop data, cukup fill missing values
        if df['Sleep Disorder'].isna().any():
            df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None')
        
        # Standardize values
        df['Sleep Disorder'] = df['Sleep Disorder'].astype(str).str.strip()
        
        # Parse Blood Pressure
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        
        features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                   'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
        
        categorical_features = ['Gender', 'Occupation', 'BMI Category']
        
        for col in categorical_features:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            features.append(col + '_encoded')
        
        self.feature_names = features
        X = df[features]
        y = df['Sleep Disorder']
        
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y)
        self.label_encoders['target'] = le_target
        self.class_labels = le_target.classes_.tolist()
        
        return X, y_encoded, df
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        
        return X_train, X_test, y_train, y_test, y_pred
    
    def predict(self, input_data):
        try:
            prediction = self.model.predict(input_data)
            prediction_proba = self.model.predict_proba(input_data)
            predicted_label = str(self.label_encoders['target'].inverse_transform(prediction)[0])
            return predicted_label, prediction_proba[0]
        except Exception as e:
            st.error(f"Error saat prediksi: {str(e)}")
            return "Unknown", np.array([0.0, 0.0, 0.0])
    
    def get_feature_importance(self):
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df

def evaluate_health_metrics(sleep_duration, quality_of_sleep, systolic_bp, diastolic_bp, 
                           heart_rate, physical_activity, daily_steps):
    """Evaluasi kondisi kesehatan"""
    results = {
        'sleep': {'status': 'baik', 'message': ''},
        'bp': {'status': 'normal', 'message': ''},
        'activity': {'status': 'cukup', 'message': ''}
    }
    
    if sleep_duration < 6.5:
        results['sleep'] = {'status': 'kurang', 'message': f'{sleep_duration} jam - Kurang dari ideal (7-9 jam)'}
    elif sleep_duration > 9:
        results['sleep'] = {'status': 'berlebih', 'message': f'{sleep_duration} jam - Terlalu banyak'}
    else:
        results['sleep'] = {'status': 'baik', 'message': f'{sleep_duration} jam - Sudah pas'}
    
    if systolic_bp >= 140 or diastolic_bp >= 90:
        results['bp'] = {'status': 'tinggi', 'message': f'{systolic_bp}/{diastolic_bp} - Tekanan darah tinggi'}
    elif systolic_bp >= 130 or diastolic_bp >= 85:
        results['bp'] = {'status': 'prehipertensi', 'message': f'{systolic_bp}/{diastolic_bp} - Agak tinggi'}
    else:
        results['bp'] = {'status': 'normal', 'message': f'{systolic_bp}/{diastolic_bp} - Normal'}
    
    if physical_activity < 30 or daily_steps < 5000:
        results['activity'] = {'status': 'kurang', 'message': 'Kurang aktif'}
    elif physical_activity >= 60 and daily_steps >= 8000:
        results['activity'] = {'status': 'baik', 'message': 'Sudah aktif'}
    else:
        results['activity'] = {'status': 'cukup', 'message': 'Cukup aktif'}
    
    return results

def show_recommendations(disorder_type):
    """Tampilkan rekomendasi berdasarkan tipe gangguan"""
    recommendations = {
        'None': {
            'title': 'Kondisi tidur kamu baik',
            'type': 'success',
            'tips': [
                'Pertahankan jadwal tidur yang teratur',
                'Tetap aktif bergerak minimal 30 menit sehari',
                'Kelola stress dengan baik',
                'Batasi konsumsi kafein setelah jam 2 siang'
            ]
        },
        'Sleep Apnea': {
            'title': 'Terdeteksi gejala Sleep Apnea',
            'type': 'error',
            'tips': [
                'Segera konsultasi ke dokter untuk penanganan lebih lanjut',
                'Jika kelebihan berat badan, coba turunkan bertahap',
                'Hindari tidur posisi telentang',
                'Kurangi atau hentikan konsumsi alkohol dan rokok',
                'Perhatikan gejala: ngorok keras, napas berhenti saat tidur, ngantuk berlebihan di siang hari'
            ]
        },
        'Insomnia': {
            'title': 'Kemungkinan mengalami Insomnia',
            'type': 'warning',
            'tips': [
                'Coba terapi CBT-I (terbukti efektif untuk insomnia)',
                'Buat rutinitas sebelum tidur yang konsisten',
                'Hindari layar HP/laptop minimal 1 jam sebelum tidur',
                'Jangan minum kopi setelah jam 2 siang',
                'Bangun di jam yang sama setiap hari termasuk weekend',
                'Jangan tidur siang jika malam susah tidur'
            ]
        },
        'default': {
            'title': 'Konsultasi ke dokter untuk hasil lebih akurat',
            'type': 'info',
            'tips': [
                'Jaga jadwal tidur yang konsisten',
                'Ciptakan suasana kamar yang nyaman untuk tidur',
                'Olahraga teratur, tapi jangan terlalu dekat dengan jam tidur',
                'Kelola stress dengan meditasi atau relaksasi'
            ]
        }
    }
    
    rec = recommendations.get(disorder_type, recommendations['default'])
    
    if rec['type'] == 'success':
        st.success(rec['title'])
    elif rec['type'] == 'error':
        st.error(rec['title'])
    elif rec['type'] == 'warning':
        st.warning(rec['title'])
    else:
        st.info(rec['title'])
    
    st.write("**Yang bisa kamu lakukan:**")
    for tip in rec['tips']:
        st.write(f"- {tip}")

@st.cache_resource
def load_model_and_data():
    predictor = SleepDisorderPredictor()
    X, y, df = predictor.load_and_prepare_data('Sleep_health_and_lifestyle_dataset.csv')
    X_train, X_test, y_train, y_test, y_pred = predictor.train_model(X, y)
    return predictor, X_train, X_test, y_train, y_test, y_pred, df

# Load model
with st.spinner('Memuat data...'):
    predictor, X_train, X_test, y_train, y_test, y_pred, df = load_model_and_data()

# Header
st.markdown('<p class="main-header">Cek Gangguan Tidur Kamu</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ["Beranda", "Prediksi", "Performa Model", "Data & Visualisasi"]
)

# HALAMAN BERANDA
if menu == "Beranda":
    st.subheader("Deteksi dini gangguan tidur pakai machine learning")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Data", f"{len(df)}")
    with col2:
        st.metric("Akurasi Model", f"{predictor.accuracy*100:.1f}%")
    with col3:
        st.metric("Algoritma", "Random Forest")
    
    st.markdown("---")
    
    st.markdown("### Tentang Aplikasi")
    st.write("""
    Aplikasi ini membantu mendeteksi kemungkinan gangguan tidur berdasarkan data kesehatan 
    dan kebiasaan harian kamu. Menggunakan algoritma Random Forest yang sudah dilatih 
    dengan ratusan data untuk prediksi yang akurat.
    
    **Apa yang bisa kamu lakukan:**
    - Cek apakah kamu berisiko gangguan tidur
    - Lihat faktor-faktor yang mempengaruhi
    - Dapatkan saran untuk tidur lebih berkualitas
    
    **Jenis gangguan yang bisa dideteksi:**
    - **Tidak ada gangguan** - Tidur kamu sehat
    - **Sleep Apnea** - Gangguan pernapasan saat tidur
    - **Insomnia** - Kesulitan tidur atau sering terbangun
    """)
    
    st.markdown("---")
    st.markdown("### Distribusi Data")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        disorder_counts = df['Sleep Disorder'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#27ae60', '#e74c3c', '#f39c12']
        disorder_counts.plot(kind='bar', ax=ax, color=colors, alpha=0.8)
        ax.set_title('Distribusi Gangguan Tidur', fontsize=13, fontweight='bold')
        ax.set_xlabel('Jenis Gangguan', fontsize=11)
        ax.set_ylabel('Jumlah', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Statistik Dataset:**")
        for disorder, count in disorder_counts.items():
            percentage = (count / len(df)) * 100
            st.write(f"{disorder}: {count} orang ({percentage:.1f}%)")

# HALAMAN PREDIKSI
elif menu == "Prediksi":
    st.subheader("Isi data di bawah untuk cek kondisi tidur kamu")
    
    with st.form("prediction_form"):
        st.markdown("#### Data Diri")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
            age = st.number_input("Usia", min_value=18, max_value=100, value=30)
        
        with col2:
            occupation = st.selectbox("Pekerjaan", sorted(df['Occupation'].unique()))
            bmi_category = st.selectbox("Kategori BMI", ["Normal", "Normal Weight", "Overweight", "Obese"])
        
        with col3:
            systolic_bp = st.number_input("Tekanan Darah Atas", min_value=90, max_value=180, value=120)
            diastolic_bp = st.number_input("Tekanan Darah Bawah", min_value=60, max_value=120, value=80)
        
        st.markdown("#### Data Tidur & Aktivitas")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sleep_duration = st.slider("Lama Tidur (jam)", 4.0, 10.0, 7.0, 0.1)
            quality_of_sleep = st.slider("Kualitas Tidur (1-10)", 1, 10, 7)
        
        with col2:
            stress_level = st.slider("Level Stress (1-10)", 1, 10, 5)
            heart_rate = st.number_input("Detak Jantung (bpm)", 50, 100, 70)
        
        with col3:
            physical_activity = st.number_input("Aktivitas Fisik (menit/hari)", 0, 180, 60, 5)
            daily_steps = st.number_input("Jumlah Langkah Harian", 0, 20000, 7000, 100)
        
        submit = st.form_submit_button("Cek Sekarang", use_container_width=True)
    
    if submit:
        # Prepare data
        input_dict = {
            'Age': age,
            'Sleep Duration': sleep_duration,
            'Quality of Sleep': quality_of_sleep,
            'Physical Activity Level': physical_activity,
            'Stress Level': stress_level,
            'Heart Rate': heart_rate,
            'Daily Steps': daily_steps,
            'Systolic_BP': systolic_bp,
            'Diastolic_BP': diastolic_bp,
            'Gender_encoded': predictor.label_encoders['Gender'].transform([gender])[0],
            'Occupation_encoded': predictor.label_encoders['Occupation'].transform([occupation])[0],
            'BMI Category_encoded': predictor.label_encoders['BMI Category'].transform([bmi_category])[0]
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # Predict
        with st.spinner('Menganalisis data kamu...'):
            predicted_disorder, probabilities = predictor.predict(input_df)
        
        predicted_disorder = str(predicted_disorder) if predicted_disorder is not None else "Unknown"
        
        st.markdown("---")
        st.markdown("### Hasil Prediksi")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if predicted_disorder == "None" or predicted_disorder == "nan":
                st.success("**Tidak ada gangguan tidur**")
            elif predicted_disorder == "Sleep Apnea":
                st.error(f"**{predicted_disorder}**")
            elif predicted_disorder == "Insomnia":
                st.warning(f"**{predicted_disorder}**")
            else:
                st.info(f"**{predicted_disorder}**")
        
        with col2:
            st.write("**Tingkat Keyakinan:**")
            class_labels_clean = [str(label) for label in predictor.class_labels]
            prob_df = pd.DataFrame({
                'Gangguan': class_labels_clean,
                'Probabilitas': probabilities * 100
            }).sort_values('Probabilitas', ascending=False)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            colors_map = {'None': '#27ae60', 'Sleep Apnea': '#e74c3c', 'Insomnia': '#f39c12'}
            colors = [colors_map.get(str(label), '#3498db') for label in prob_df['Gangguan']]
            
            # Convert to list untuk avoid matplotlib error
            labels_list = [str(x) for x in prob_df['Gangguan'].values]
            values_list = [float(x) for x in prob_df['Probabilitas'].values]
            
            bars = ax.barh(range(len(labels_list)), values_list, color=colors, alpha=0.8)
            ax.set_yticks(range(len(labels_list)))
            ax.set_yticklabels(labels_list)
            ax.set_xlabel('Probabilitas (%)', fontsize=10)
            
            for i, val in enumerate(values_list):
                ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Rekomendasi
        st.markdown("### Saran untuk Kamu")
        show_recommendations(predicted_disorder)
        
        st.markdown("---")
        
        # Health metrics
        st.markdown("### Kondisi Kesehatan Kamu")
        health_eval = evaluate_health_metrics(
            sleep_duration, quality_of_sleep, systolic_bp, diastolic_bp,
            heart_rate, physical_activity, daily_steps
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Tidur**")
            if health_eval['sleep']['status'] == 'baik':
                st.success(health_eval['sleep']['message'])
            else:
                st.warning(health_eval['sleep']['message'])
            
            if quality_of_sleep < 6:
                st.warning(f"Kualitas: {quality_of_sleep}/10 - Perlu ditingkatkan")
            else:
                st.success(f"Kualitas: {quality_of_sleep}/10")
        
        with col2:
            st.write("**Kardiovaskular**")
            if health_eval['bp']['status'] == 'normal':
                st.success(health_eval['bp']['message'])
            elif health_eval['bp']['status'] == 'prehipertensi':
                st.warning(health_eval['bp']['message'])
            else:
                st.error(health_eval['bp']['message'])
            
            if 60 <= heart_rate <= 90:
                st.success(f"Detak jantung: {heart_rate} bpm")
            else:
                st.warning(f"Detak jantung: {heart_rate} bpm")
        
        with col3:
            st.write("**Aktivitas Fisik**")
            if health_eval['activity']['status'] == 'baik':
                st.success(f"{physical_activity} menit/hari")
            elif health_eval['activity']['status'] == 'kurang':
                st.error(f"{physical_activity} menit/hari - Kurang aktif")
            else:
                st.info(f"{physical_activity} menit/hari")
            
            if daily_steps >= 8000:
                st.success(f"{daily_steps:,} langkah")
            elif daily_steps < 5000:
                st.error(f"{daily_steps:,} langkah - Kurang")
            else:
                st.info(f"{daily_steps:,} langkah")

# HALAMAN PERFORMA MODEL
elif menu == "Performa Model":
    st.subheader("Seberapa Akurat Prediksinya?")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Akurasi", f"{predictor.accuracy*100:.1f}%")
    with col2:
        st.metric("Jumlah Trees", "200")
    with col3:
        st.metric("Max Depth", "15")
    with col4:
        st.metric("Test Size", "20%")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=predictor.class_labels,
                   yticklabels=predictor.class_labels)
        ax.set_xlabel('Prediksi', fontsize=11)
        ax.set_ylabel('Aktual', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.write("**Fitur Paling Berpengaruh**")
        importance_df = predictor.get_feature_importance().head(10)
        importance_df['Feature_Clean'] = importance_df['Feature'].str.replace('_encoded', '').str.replace('_', ' ').str.title()
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(importance_df['Feature_Clean'], importance_df['Importance'], color='steelblue')
        ax.set_xlabel('Importance', fontsize=11)
        ax.invert_yaxis()
        
        for i, val in enumerate(importance_df['Importance']):
            ax.text(val + 0.003, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    st.write("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=predictor.class_labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    
    st.dataframe(
        report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score'])
                       .format("{:.3f}", subset=['precision', 'recall', 'f1-score'])
                       .format("{:.0f}", subset=['support']),
        use_container_width=True
    )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Kenapa Random Forest?**")
        st.write(f"""
        - Akurasi tinggi: {predictor.accuracy*100:.1f}%
        - Bisa handle data kompleks
        - Robust terhadap outlier
        - Bisa identifikasi fitur penting
        - Tidak mudah overfit
        """)
    
    with col2:
        st.write("**Fitur Paling Penting:**")
        top_5 = predictor.get_feature_importance().head(5)
        for idx, row in top_5.iterrows():
            feature_clean = row['Feature'].replace('_encoded', '').replace('_', ' ').title()
            st.write(f"- {feature_clean}: {row['Importance']:.4f}")

# HALAMAN VISUALISASI DATA
else:
    st.subheader("Eksplorasi Data")
    
    tab1, tab2, tab3 = st.tabs(["Distribusi Fitur", "Korelasi", "Pola Tidur"])
    
    with tab1:
        numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                           'Stress Level', 'Heart Rate', 'Daily Steps']
        
        selected_feature = st.selectbox("Pilih fitur:", numeric_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df[selected_feature].hist(bins=25, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel(selected_feature, fontsize=11)
            ax.set_ylabel('Frekuensi', fontsize=11)
            ax.axvline(df[selected_feature].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(df[selected_feature].median(), color='green', linestyle='--', linewidth=2, label='Median')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write(f"**Statistik {selected_feature}**")
            stats_df = df[selected_feature].describe().round(2)
            st.dataframe(stats_df, use_container_width=True)
            
            st.write("**Per Gangguan Tidur**")
            disorder_stats = df.groupby('Sleep Disorder')[selected_feature].agg(['mean', 'std']).round(2)
            st.dataframe(disorder_stats, use_container_width=True)
    
    with tab2:
        numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                       'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic_BP', 'Diastolic_BP']
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            df.boxplot(column='Sleep Duration', by='Sleep Disorder', ax=ax)
            ax.set_xlabel('Gangguan Tidur', fontsize=11)
            ax.set_ylabel('Durasi Tidur (jam)', fontsize=11)
            plt.suptitle('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            df.boxplot(column='Stress Level', by='Sleep Disorder', ax=ax)
            ax.set_xlabel('Gangguan Tidur', fontsize=11)
            ax.set_ylabel('Level Stress', fontsize=11)
            plt.suptitle('')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# Footer
st.markdown("---")
st.caption("Disclaimer: Hasil prediksi ini hanya untuk referensi awal. Untuk diagnosis yang akurat, konsultasi dengan dokter.")
