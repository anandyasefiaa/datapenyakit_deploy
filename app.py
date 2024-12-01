import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi Preprocessing
def preprocess_data(data):
    # Identifikasi missing values
    missing_proportions = data.isnull().mean()
    high_missing_cols = missing_proportions[missing_proportions > 0.2]
    low_missing_cols = missing_proportions[(missing_proportions > 0) & (missing_proportions <= 0.2)]

    # Tangani kolom dengan missing value tinggi
    for col in high_missing_cols.index:
        if col in data.columns:
            sample_values = data[col].dropna().sample(data[col].isnull().sum(), replace=True).values
            data.loc[data[col].isnull(), col] = sample_values

    # Tangani kolom dengan missing value rendah
    imputer_mean = SimpleImputer(strategy='mean')
    imputer_mode = SimpleImputer(strategy='most_frequent')

    for col in low_missing_cols.index:
        if col in data.columns:
            if data[col].dtype in ['float64', 'int64']:
                data[col] = imputer_mean.fit_transform(data[[col]]).flatten()
            else:
                data[col] = imputer_mode.fit_transform(data[[col]]).flatten()

    # Ubah tipe data objek menjadi string
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].astype(str)

    # One-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Konversi boolean ke integer
    bool_columns = data_encoded.columns[data_encoded.dtypes == bool]
    data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

    return data_encoded

# Fungsi Utama Streamlit
def main():
    st.set_page_config(
        page_title="Klasifikasi Penyakit Gagal Ginjal Kronis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Klasifikasi Penyakit Gagal Ginjal Kronis")

    uploaded_file = st.sidebar.file_uploader("Upload Dataset CSV", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Silakan upload file dataset terlebih dahulu.")
        return

    page = st.sidebar.selectbox("Navigasi", ["Data", "Preprocessing", "Modeling", "Evaluasi"])

    if page == "Data":
        st.header("Data Asli")
        st.dataframe(data)
        st.write(f"Jumlah Baris: {data.shape[0]}, Jumlah Kolom: {data.shape[1]}")
        st.write("Informasi Data:")
        st.text(data.info())

    elif page == "Preprocessing":
        st.header("Preprocessing Data")
        st.subheader("Data Sebelum Preprocessing")
        st.dataframe(data)

        # Preprocessing
        preprocessed_data = preprocess_data(data)

        st.subheader("Data Setelah Preprocessing")
        st.dataframe(preprocessed_data)
        
        # Identifikasi missing values
        missing_proportions = data.isnull().mean()
        high_missing_cols = missing_proportions[missing_proportions > 0.2]
        low_missing_cols = missing_proportions[(missing_proportions > 0) & (missing_proportions <= 0.2)]

        # Tampilkan Missing Value Tinggi dan Rendah
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Kolom dengan Missing Value Tinggi (>20%)")
            if not high_missing_cols.empty:
                st.write(high_missing_cols.rename("Proporsi Missing Value"))
            else:
                st.write("Tidak ada kolom dengan missing value tinggi.")

        with col2:
            st.subheader("Kolom dengan Missing Value Rendah (0-20%)")
            if not low_missing_cols.empty:
                st.write(low_missing_cols.rename("Proporsi Missing Value"))
            else:
                st.write("Tidak ada kolom dengan missing value rendah.")

        # Korelasi dan Seleksi Fitur
        st.subheader("Seleksi Fitur Berdasarkan Korelasi")
        if 'classification_notckd' in preprocessed_data.columns:
            corr_matrix = preprocessed_data.corr()
            Dependent_corr = corr_matrix['classification_notckd']
            Imp_features = Dependent_corr[Dependent_corr.abs() > 0.4].index.tolist()

            st.write("Fitur yang Dipilih:")
            st.write(Imp_features if Imp_features else "Tidak ada fitur yang memenuhi syarat korelasi.")
        else:
            st.error("Kolom target 'classification_notckd' tidak ditemukan.")

        st.session_state['preprocessed_data'] = preprocessed_data
        st.session_state['Imp_features'] = Imp_features if Imp_features else []

    elif page == "Modeling":
        st.header("Modeling")
        if 'preprocessed_data' in st.session_state and 'Imp_features' in st.session_state:
            preprocessed_data = st.session_state['preprocessed_data']
            Imp_features = st.session_state['Imp_features']

            if 'classification_notckd' in preprocessed_data.columns and Imp_features:
                # Memisahkan data fitur (X) dan target (y)
                X = preprocessed_data[Imp_features]
                y = preprocessed_data['classification_notckd']

                # Split Data menjadi Train dan Test
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Tambahkan random_state untuk konsistensi

                st.subheader("Pilih Model")
                model_choice = st.selectbox("Model", ["Naive Bayes", "Decision Tree"])

                if model_choice == "Naive Bayes":
                    model = GaussianNB()
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier()

                # Training Model
                model.fit(X_train, y_train)
                
                # Hasil Pelatihan pada Data Latih
                st.subheader("Hasil Pelatihan Model")
                y_pred_train = model.predict(X_train)
                train_accuracy = accuracy_score(y_train, y_pred_train)
                st.write(f"Accuracy pada Data Latih: {train_accuracy:.4f}")

                # Menyimpan model dan data untuk evaluasi
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.success(f"Model {model_choice} berhasil dilatih!")
            else:
                st.error("Data preprocessing belum selesai atau fitur penting tidak ditemukan.")
        else:
            st.error("Lakukan preprocessing terlebih dahulu.")

    elif page == "Evaluasi":
        st.header("Evaluasi Model")
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            # Evaluasi pada Data Uji
            st.subheader("Hasil Evaluasi Model")
            y_pred = model.predict(X_test)  # Pastikan menggunakan X_test untuk prediksi
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy pada Data Uji: {accuracy:.4f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))
        else:
            st.error("Latih model terlebih dahulu pada halaman Modeling.")

if __name__ == '__main__':
    main()
