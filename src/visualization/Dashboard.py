import pandas as pd
import streamlit as st
import altair as alt


from chatbot import mostrar_chatbot


st.set_page_config(
    page_title="Edu AI Risk Dashboard",
    layout="wide"
)

# =========================
# Estilos
# =========================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 80px;
        font-weight: 900;
        color: #ffffff;
        margin-bottom: 0px;
        text-align: center;
        letter-spacing: 2px;
    }

    .subtitle {
        font-size: 30px;
        color: #ffffff;
        margin-bottom: 25px;
        text-align: center;
    }

    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        border-left: 7px solid #2563eb;
    }

    .metric-card-risk {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
        border-left: 7px solid #dc2626;
    }

    .metric-title {
        font-size: 14px;
        color: #6b7280;
        font-weight: 600;
    }

    .metric-value {
        font-size: 30px;
        color: #111827;
        font-weight: 800;
    }

    .metric-delta {
        font-size: 13px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">EDU AI RISK</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Dashboard para detectar alumnos con riesgo de reprobar y apoyar decisiones académicas.</p>',
    unsafe_allow_html=True
)

# =========================
# Cargar datos
# =========================

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/student-mat-prediction.csv")


df = load_data().copy()


# =========================
# Limpieza y columnas auxiliares
# =========================

def normalizar_prediccion(valor):
    valor = str(valor).strip().lower()

    if "reprob" in valor or "reprueb" in valor or "reprue" in valor:
        return "Reprueba"
    elif "aprueb" in valor or "aprob" in valor:
        return "Aprueba"
    else:
        return "Sin dato"


df["estado_g1"] = df["G1"].apply(lambda x: "Aprueba" if x >= 10 else "Reprueba")
df["estado_g2"] = df["G2"].apply(lambda x: "Aprueba" if x >= 10 else "Reprueba")
df["prediccion_modelo_limpia"] = df["prediccion_modelo"].apply(normalizar_prediccion)

df["alumno_id"] = df.index + 1

# Asegurar que la probabilidad esté en escala 0 a 1
if df["probabilidad_reprueba"].max() > 1:
    df["probabilidad_reprueba_base"] = df["probabilidad_reprueba"] / 100
else:
    df["probabilidad_reprueba_base"] = df["probabilidad_reprueba"]

df["probabilidad_reprueba_pct"] = df["probabilidad_reprueba_base"] * 100


def clasificar_riesgo(prob):
    if prob >= 0.70:
        return "Alto"
    elif prob >= 0.40:
        return "Medio"
    else:
        return "Bajo"


df["nivel_riesgo"] = df["probabilidad_reprueba_base"].apply(clasificar_riesgo)

studytime_labels = {
    1: "< 2 horas",
    2: "2 a 5 horas",
    3: "5 a 10 horas",
    4: "> 10 horas"
}

df["studytime_label"] = df["studytime"].map(studytime_labels)

df["failures_group"] = df["failures"].apply(
    lambda x: "3+ fracasos" if x >= 3 else f"{x} fracasos"
)


# =========================
# Colores
# =========================

colores_estado = alt.Scale(
    domain=["Aprueba", "Reprueba", "Sin dato"],
    range=["#16a34a", "#dc2626", "#6b7280"]
)

colores_riesgo = alt.Scale(
    domain=["Bajo", "Medio", "Alto"],
    range=["#16a34a", "#f59e0b", "#dc2626"]
)


# =========================
# Función para tarjetas
# =========================

def metric_card(title, value, delta="", risk=False):
    css_class = "metric-card-risk" if risk else "metric-card"

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True
    )





# =========================
# Métricas principales
# =========================

total_estudiantes = len(df)

aprobados_g2 = (df["G2"] >= 10).sum()
reprobados_g2 = (df["G2"] < 10).sum()
promedio_g2 = df["G2"].mean()

alumnos_riesgo_reprobar = (
    df["prediccion_modelo_limpia"] == "Reprueba"
).sum()

riesgo_alto = (df["nivel_riesgo"] == "Alto").sum()
riesgo_medio = (df["nivel_riesgo"] == "Medio").sum()
riesgo_bajo = (df["nivel_riesgo"] == "Bajo").sum()

st.subheader("Resumen de los estudiantes")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    metric_card(
        "Total estudiantes",
        total_estudiantes,
        "Registros analizados"
    )

with col2:
    metric_card(
        "Aprobados G2",
        aprobados_g2,
        f"{aprobados_g2 / total_estudiantes:.1%} del total"
    )

with col3:
    metric_card(
        "Reprobados G2",
        reprobados_g2,
        f"{reprobados_g2 / total_estudiantes:.1%} del total",
        risk=True
    )

with col4:
    metric_card(
        "Promedio G2",
        round(promedio_g2, 2),
        "Calificación promedio"
    )

with col5:
    metric_card(
        "Riesgo de reprobar",
        alumnos_riesgo_reprobar,
        f"{alumnos_riesgo_reprobar / total_estudiantes:.1%} predichos por el modelo",
        risk=True
    )

st.write("")

col1, col2, col3 = st.columns(3)

col1.metric("Riesgo alto", riesgo_alto, f"{riesgo_alto / total_estudiantes:.1%}")
col2.metric("Riesgo medio", riesgo_medio, f"{riesgo_medio / total_estudiantes:.1%}")
col3.metric("Riesgo bajo", riesgo_bajo, f"{riesgo_bajo / total_estudiantes:.1%}")

# =========================
# Visualizaciones exploratorias
# =========================

st.subheader("Visualizaciones exploratorias")

col1, col2 = st.columns(2)

with col1:
    estado_g2_counts = (
        df["estado_g2"]
        .value_counts()
        .reindex(["Aprueba", "Reprueba"])
        .fillna(0)
        .reset_index()
    )

    estado_g2_counts.columns = ["Estado G2", "Cantidad"]

    chart = (
        alt.Chart(estado_g2_counts)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Estado G2:N", title="Estado"),
            y=alt.Y("Cantidad:Q", title="Cantidad"),
            color=alt.Color("Estado G2:N", scale=colores_estado, legend=None),
            tooltip=["Estado G2", "Cantidad"]
        )
        .properties(
            title="Aprobados vs reprobados según G2",
            height=330
        )
    )

    text = chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=14
    ).encode(
        text="Cantidad:Q"
    )

    st.altair_chart(chart + text, use_container_width=True)

with col2:
    absences_by_estado_g2 = (
        df.groupby("estado_g2")["absences"]
        .mean()
        .reindex(["Aprueba", "Reprueba"])
        .fillna(0)
        .reset_index()
    )

    absences_by_estado_g2.columns = ["Estado G2", "Ausencias promedio"]

    chart = (
        alt.Chart(absences_by_estado_g2)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X("Estado G2:N", title="Estado"),
            y=alt.Y("Ausencias promedio:Q", title="Ausencias promedio"),
            color=alt.Color("Estado G2:N", scale=colores_estado, legend=None),
            tooltip=["Estado G2", alt.Tooltip("Ausencias promedio:Q", format=".2f")]
        )
        .properties(
            title="Ausencias promedio por estado según G2",
            height=330
        )
    )

    st.altair_chart(chart, use_container_width=True)

# =========================
# Distribución de G2
# =========================

st.subheader("Distribución de calificaciones G2")

g2_counts = (
    df["G2"]
    .value_counts()
    .sort_index()
    .reset_index()
)

g2_counts.columns = ["G2", "Cantidad"]

chart = (
    alt.Chart(g2_counts)
    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
    .encode(
        x=alt.X("G2:O", title="Calificación G2"),
        y=alt.Y("Cantidad:Q", title="Cantidad de estudiantes"),
        color=alt.Color(
            "Cantidad:Q",
            scale=alt.Scale(scheme="blues"),
            legend=None
        ),
        tooltip=["G2", "Cantidad"]
    )
    .properties(
        title="Distribución de calificaciones en G2",
        height=350
    )
)

st.altair_chart(chart, use_container_width=True)

# =========================
# Tiempo de estudio vs G2
# =========================

st.subheader("Tiempo de estudio semanal vs nota promedio G2")

g2_by_studytime = (
    df.groupby("studytime_label")["G2"]
    .mean()
    .reindex(["< 2 horas", "2 a 5 horas", "5 a 10 horas", "> 10 horas"])
    .reset_index()
)

g2_by_studytime.columns = ["Tiempo de estudio", "Promedio G2"]

chart = (
    alt.Chart(g2_by_studytime)
    .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
    .encode(
        x=alt.X(
            "Tiempo de estudio:N",
            sort=["< 2 horas", "2 a 5 horas", "5 a 10 horas", "> 10 horas"],
            title="Tiempo de estudio semanal"
        ),
        y=alt.Y("Promedio G2:Q", title="Promedio G2"),
        color=alt.Color(
            "Promedio G2:Q",
            scale=alt.Scale(scheme="tealblues"),
            legend=None
        ),
        tooltip=[
            "Tiempo de estudio",
            alt.Tooltip("Promedio G2:Q", format=".2f")
        ]
    )
    .properties(
        title="Promedio G2 por tiempo de estudio semanal",
        height=350
    )
)

st.altair_chart(chart, use_container_width=True)




col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicción por número de fracasos previos")

    failures_prediccion = pd.crosstab(
        df["failures_group"],
        df["prediccion_modelo_limpia"]
    )

    failures_prediccion = failures_prediccion.reindex(
        ["0 fracasos", "1 fracasos", "2 fracasos", "3+ fracasos"]
    ).fillna(0)

    failures_prediccion = failures_prediccion.reindex(
        columns=["Aprueba", "Reprueba"],
        fill_value=0
    )

    failures_prediccion = failures_prediccion.reset_index()

    failures_prediccion = failures_prediccion.melt(
        id_vars="failures_group",
        value_vars=["Aprueba", "Reprueba"],
        var_name="Predicción",
        value_name="Cantidad"
    )

    chart = (
        alt.Chart(failures_prediccion)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "failures_group:N",
                sort=["0 fracasos", "1 fracasos", "2 fracasos", "3+ fracasos"],
                title="Fracasos previos"
            ),
            y=alt.Y("Cantidad:Q", title="Cantidad"),
            color=alt.Color("Predicción:N", scale=colores_estado),
            xOffset="Predicción:N",
            tooltip=["failures_group", "Predicción", "Cantidad"]
        )
        .properties(
            title="Predicción del modelo según fracasos previos",
            height=350
        )
    )

    st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("Progresión de estudiantes aprobados y reprobados")

    progreso_estudiantes = pd.DataFrame(
        {
            "Etapa": ["G1", "G2", "Predicción modelo"],
            "Aprueba": [
                (df["G1"] >= 10).sum(),
                (df["G2"] >= 10).sum(),
                (df["prediccion_modelo_limpia"] == "Aprueba").sum()
            ],
            "Reprueba": [
                (df["G1"] < 10).sum(),
                (df["G2"] < 10).sum(),
                (df["prediccion_modelo_limpia"] == "Reprueba").sum()
            ]
        }
    )

    progreso_long = progreso_estudiantes.melt(
        id_vars="Etapa",
        value_vars=["Aprueba", "Reprueba"],
        var_name="Estado",
        value_name="Cantidad"
    )

    chart = (
        alt.Chart(progreso_long)
        .mark_line(point=True, strokeWidth=4)
        .encode(
            x=alt.X(
                "Etapa:N",
                sort=["G1", "G2", "Predicción modelo"],
                title="Etapa"
            ),
            y=alt.Y("Cantidad:Q", title="Cantidad de estudiantes"),
            color=alt.Color("Estado:N", scale=colores_estado),
            tooltip=["Etapa", "Estado", "Cantidad"]
        )
        .properties(
            title="Evolución G1 → G2 → Predicción del modelo",
            height=350
        )
    )

    st.altair_chart(chart, use_container_width=True)


st.subheader("Análisis de riesgo académico")

# Distribución por nivel de riesgo
riesgo_counts = (
df["nivel_riesgo"]
.value_counts()
.reindex(["Bajo", "Medio", "Alto"])
.fillna(0)
.reset_index()
)

riesgo_counts.columns = ["Nivel de riesgo", "Cantidad"]

chart = (
alt.Chart(riesgo_counts)
.mark_arc(innerRadius=70)
.encode(
    theta=alt.Theta("Cantidad:Q"),
    color=alt.Color("Nivel de riesgo:N", scale=colores_riesgo),
    tooltip=["Nivel de riesgo", "Cantidad"]
)
.properties(
    title="Distribución por nivel de riesgo",
    height=350
)
)

st.altair_chart(chart, use_container_width=True)


col1, col2 = st.columns(2)

with col1:
    riesgo_studytime = pd.crosstab(
        df["studytime_label"],
        df["nivel_riesgo"]
    )

    riesgo_studytime = riesgo_studytime.reindex(
        ["< 2 horas", "2 a 5 horas", "5 a 10 horas", "> 10 horas"]
    ).fillna(0)

    riesgo_studytime = riesgo_studytime.reindex(
        columns=["Bajo", "Medio", "Alto"],
        fill_value=0
    )

    riesgo_studytime = riesgo_studytime.reset_index().melt(
        id_vars="studytime_label",
        value_vars=["Bajo", "Medio", "Alto"],
        var_name="Nivel de riesgo",
        value_name="Cantidad"
    )

    chart = (
        alt.Chart(riesgo_studytime)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X(
                "studytime_label:N",
                sort=["< 2 horas", "2 a 5 horas", "5 a 10 horas", "> 10 horas"],
                title="Tiempo de estudio"
            ),
            y=alt.Y("Cantidad:Q", title="Cantidad"),
            color=alt.Color("Nivel de riesgo:N", scale=colores_riesgo),
            xOffset="Nivel de riesgo:N",
            tooltip=["studytime_label", "Nivel de riesgo", "Cantidad"]
        )
        .properties(
            title="Nivel de riesgo por tiempo de estudio",
            height=350
        )
    )

    st.altair_chart(chart, use_container_width=True)

with col2:
    top_riesgo = (
        df.sort_values("probabilidad_reprueba_base", ascending=False)
        .head(15)
        .copy()
    )

    top_riesgo["Alumno"] = "Alumno " + top_riesgo["alumno_id"].astype(str)

    chart = (
        alt.Chart(top_riesgo)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Alumno:N", sort="-y", title="Alumno"),
            y=alt.Y(
                "probabilidad_reprueba_pct:Q",
                title="Probabilidad de reprobar (%)"
            ),
            color=alt.Color("nivel_riesgo:N", scale=colores_riesgo),
            tooltip=[
                "Alumno",
                "G1",
                "G2",
                "absences",
                "failures",
                "studytime_label",
                "nivel_riesgo",
                alt.Tooltip("probabilidad_reprueba_pct:Q", format=".1f")
            ]
        )
        .properties(
            title="Top 15 alumnos con mayor probabilidad de reprobar",
            height=350
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # =========================
    # Tabla de alumnos prioritarios
    # =========================

st.subheader("Alumnos prioritarios para intervención")

alumnos_prioritarios = (
    df.sort_values("probabilidad_reprueba_base", ascending=False)
    .loc[
        :,
        [
            "alumno_id",
            "G1",
            "G2",
            "absences",
            "failures",
            "studytime_label",
            "prediccion_modelo_limpia",
            "nivel_riesgo",
            "probabilidad_reprueba_pct"
        ]
    ]
    .head(25)
)

alumnos_prioritarios = alumnos_prioritarios.rename(
    columns={
        "alumno_id": "Alumno",
        "absences": "Ausencias",
        "failures": "Fracasos previos",
        "studytime_label": "Tiempo de estudio",
        "prediccion_modelo_limpia": "Predicción modelo",
        "nivel_riesgo": "Nivel de riesgo",
        "probabilidad_reprueba_pct": "Probabilidad de reprobar (%)"
    }
)

alumnos_prioritarios["Probabilidad de reprobar (%)"] = alumnos_prioritarios[
    "Probabilidad de reprobar (%)"
].round(2)

st.dataframe(
    alumnos_prioritarios,
    use_container_width=True,
    hide_index=True
    )

st.markdown("---")
st.header("Chatbot")
mostrar_chatbot()