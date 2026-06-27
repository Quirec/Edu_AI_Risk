

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    make_scorer
)


# ==============================
# 1. Rutas del proyecto
# ==============================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = "data\processed\student-mat-clean.csv"
MODEL_PATH = "models\modelo_regresion_logistica.pkl"
METRICS_PATH = "reports\metricas_regresion_logistica.csv"
REPORT_PATH =  "reports\ reporte_entrenamiento_regresion_logistica.txt"
PREDICTIONS_PATH= "data\processed\student-mat-prediction.csv"





# ==============================
# 2. Cargar dataset
# ==============================

df = pd.read_csv(DATA_PATH)

print("Dataset cargado correctamente")
print(df.head())
print(df.columns)


# ==============================
# 3. Seleccionar features y variable objetivo
# ==============================

features = ["G1", "G2", "absences", "studytime", "failures"]

X = df[features]

# aprueba ya está en binario:
# aprueba = 1
# reprueba = 0
#
# Pero como queremos detectar alumnos que reprueban:
# reprobado = 1
# aprueba = 0

y = 1 - df["aprueba"]

print("\nDistribución de clases:")
print(y.value_counts())

print("\nInterpretación:")
print("0 = Aprueba")
print("1 = Reprueba")


# ==============================
# 4. Dividir entrenamiento y prueba
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==============================
# 5. Pipeline del modelo
# ==============================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("modelo", LogisticRegression(max_iter=1000, class_weight="balanced"))
])


# ==============================
# 6. Hiperparámetros
# ==============================

param_grid = {
    "modelo__C": [0.01, 0.1, 1, 10, 100],
    "modelo__penalty": ["l1", "l2"],
    "modelo__solver": ["liblinear"]
}


scoring = {
    "Accuracy": "accuracy",
    "F1-score": make_scorer(f1_score, zero_division=0),
    "Precision": make_scorer(precision_score, zero_division=0),
    "Recall": make_scorer(recall_score, zero_division=0)
}


grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring=scoring,
    refit="F1-score",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)


# ==============================
# 7. Entrenar modelo
# ==============================

grid_search.fit(X_train, y_train)

print("\nMejores hiperparámetros:")
print(grid_search.best_params_)

print("\nMejor F1-score en validación cruzada:")
print(grid_search.best_score_)

# ==============================
# Evaluar modelo final
# ==============================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

# Matriz de confusión
matriz = confusion_matrix(y_test, y_pred)

matriz_df = pd.DataFrame(
    matriz,
    index=["Real Aprueba", "Real Reprueba"],
    columns=["Predice Aprueba", "Predice Reprueba"]
)

# Reporte de clasificación
reporte_clasificacion = classification_report(
    y_test,
    y_pred,
    target_names=["Aprueba", "Reprueba"],
    zero_division=0
)

# ==============================
# Imprimir resultados
# ==============================

print("\nMejores hiperparámetros:")
print(grid_search.best_params_)

print("\nMejor F1-score en validación cruzada:")
print(grid_search.best_score_)

print("\nMétricas en conjunto de prueba:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

print("\nMatriz de confusión:")
print(matriz_df)

print("\nReporte de clasificación:")
print(reporte_clasificacion)

# ==============================
# Guardar reporte completo en TXT
# ==============================

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("REPORTE DE ENTRENAMIENTO - REGRESIÓN LOGÍSTICA\n")
    f.write("=" * 60)
    f.write("\n\n")

    f.write("Features utilizadas:\n")
    f.write(str(features))
    f.write("\n\n")

    f.write("Distribución de clases:\n")
    f.write(str(y.value_counts()))
    f.write("\n\n")

    f.write("Interpretación de clases:\n")
    f.write("0 = Aprueba\n")
    f.write("1 = Reprueba\n")
    f.write("\n")

    f.write("Mejores hiperparámetros:\n")
    f.write(str(grid_search.best_params_))
    f.write("\n\n")

    f.write("Mejor F1-score en validación cruzada:\n")
    f.write(str(grid_search.best_score_))
    f.write("\n\n")

    f.write("Métricas en conjunto de prueba:\n")
    f.write(f"Accuracy:  {accuracy:.4f}\n")
    f.write(f"F1-score:  {f1:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall:    {recall:.4f}\n")
    f.write("\n")

    f.write("Matriz de confusión:\n")
    f.write(matriz_df.to_string())
    f.write("\n\n")

    f.write("Reporte de clasificación:\n")
    f.write(reporte_clasificacion)
    f.write("\n")

print(f"\nReporte completo guardado en: {REPORT_PATH}")

# ==============================
# 12. Crear nuevo dataset con predicciones
# ==============================


# Usamos las mismas features con las que entrenaste el modelo
X_total = df[features]

# Predicción para todos los alumnos
predicciones = best_model.predict(X_total)

# Probabilidades
probabilidades = best_model.predict_proba(X_total)

# Crear copia del dataset original
df_predicciones = df.copy()

# Como usamos:
# y = 1 - df["aprueba"]
# entonces:
# 0 = Aprueba
# 1 = Reprueba

df_predicciones["prediccion_reprobado"] = predicciones

df_predicciones["prediccion_modelo"] = df_predicciones["prediccion_reprobado"].map({
    0: "Aprueba",
    1: "Reprueba"
})

# Probabilidades
# Columna 0 = probabilidad de Aprueba
# Columna 1 = probabilidad de Reprueba
df_predicciones["probabilidad_aprueba"] = probabilidades[:, 0]
df_predicciones["probabilidad_reprueba"] = probabilidades[:, 1]

# Comparar contra el resultado real
df_predicciones["resultado_real"] = df_predicciones["aprueba"].map({
    1: "Aprueba",
    0: "Reprueba"
})

df_predicciones["acierto_modelo"] = (
    df_predicciones["resultado_real"] == df_predicciones["prediccion_modelo"]
)

# Guardar nuevo dataset
df_predicciones.to_csv(PREDICTIONS_PATH, index=False)

print(f"\nDataset con predicciones guardado en: {PREDICTIONS_PATH}")

print("\nPrimeras filas del dataset con predicciones:")
print(df_predicciones[[
    "G1",
    "G2",
    "resultado_real",
    "prediccion_modelo",
    "probabilidad_aprueba",
    "probabilidad_reprueba",
    "acierto_modelo"
]].head())