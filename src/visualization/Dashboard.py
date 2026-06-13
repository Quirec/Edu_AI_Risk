import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Edu AI Risk Dashboard",
    layout="wide"
)

st.title("Edu AI Risk")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/student-mat-clean.csv")



df = load_data()

tab_dashboard, tab_chatbot = st.tabs([" Dashboard", "Chatbot"])


with tab_dashboard:

    # =========================
    # Métricas principales
    # =========================

    total_estudiantes = len(df)
    aprobados = (df["G3"] >= 10).sum()
    reprobados = (df["G3"] < 10).sum()
    promedio_g3 = df["G3"].mean()

    st.subheader("Resumen del dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total estudiantes", total_estudiantes)
    col2.metric("Aprobados", aprobados, f"{aprobados / total_estudiantes:.1%}")
    col3.metric("Reprobados", reprobados, f"{reprobados / total_estudiantes:.1%}")
    col4.metric("Promedio G3", round(promedio_g3, 2))

    # =========================
    # Visualizaciones exploratorias
    # =========================

    st.subheader("Visualizaciones exploratorias")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Aprobados vs reprobados")
        estado_counts = df["resultado"].value_counts()
        st.bar_chart(estado_counts)

    with col2:
        st.write("Ausencias promedio por estado")
        absences_by_estado = df.groupby("resultado")["absences"].mean()
        st.bar_chart(absences_by_estado)

    # =========================
    # Distribución de G3
    # =========================

    st.subheader("Distribución de calificaciones finales G3")

    g3_counts = df["G3"].value_counts().sort_index()
    st.bar_chart(g3_counts)

    # =========================
    # Tiempo de estudio vs G3
    # =========================

    st.subheader("Tiempo de estudio semanal vs nota final promedio")

    studytime_labels = {
        1: "< 2 horas",
        2: "2 a 5 horas",
        3: "5 a 10 horas",
        4: "> 10 horas"
    }

    df["studytime_label"] = df["studytime"].map(studytime_labels)

    g3_by_studytime = (
        df.groupby("studytime_label")["G3"]
        .mean()
        .reindex(["< 2 horas", "2 a 5 horas", "5 a 10 horas", "> 10 horas"])
    )

    st.bar_chart(g3_by_studytime)

    # =========================
    # Fracasos previos vs estado final
    # =========================

    col1, col2 = st.columns(2)


    with col2:
        st.subheader("Progresión de notas G1 → G2 → G3")

        progress = df.groupby("resultado")[["G1", "G2", "G3"]].mean().T

        st.line_chart(progress)

    # =========================
    # Tabla de datos
    # =========================

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(50), use_container_width=True)

with tab_chatbot:
    st.header("Chatbot")

    st.write("Aquí se integrará el chatbot del proyecto.")