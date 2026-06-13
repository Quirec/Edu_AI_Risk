import os
import pandas as pd


df_clean = pd.read_csv("data/raw/student-mat.csv", sep=";")


# Crear nuevas columnas
df_clean["aprueba"] = (df_clean["G3"] >= 10).astype(int)
df_clean["resultado"] = df_clean["aprueba"].map({
    1: "Aprueba",
    0: "Reprueba"
})

# Crear un NUEVO archivo CSV
df_clean.to_csv("student-mat-clean.csv", index=False)

print("Archivo nuevo creado: student-mat-clean.csv")


os.makedirs("data/processed", exist_ok=True)

df_clean.to_csv("data/processed/student-mat-clean.csv", index=False)

print("Archivo nuevo creado en data/processed/student-mat-clean.csv")

