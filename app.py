from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import os

app = Flask(__name__)

# Cargar modelo y scaler
model = load_model("modelo_entrenado.keras")
scaler = joblib.load("scaler.pkl")

# Cargar columnas del CSV de referencia
columns = pd.read_csv("datos_preinfeccion_limpios.csv", sep=";", encoding="ISO-8859-1").drop("ILQ", axis=1).columns.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Recoger y transformar inputs del formulario
            inputs = {
                "EDAD": float(request.form["EDAD"]),
                "Sexo": float(request.form["Sexo"]),
                "PROCED": 1.0 if request.form["PROCED"] == "1" else 2.0,
                "Proc.Duración": float(request.form["Proc.Duración"]),
                "Proc.Esreintervención": float(request.form["Proc.Esreintervención"]),
                "TipoIntervención": float(request.form["TipoIntervención"]),
                "Quirófano_1.0": 1.0 if request.form["QUIROFANO"] == "1" else 0.0,
                "Proc.ASA_1": 1.0 if request.form["ASA"] == "1" else 0.0,
                "ValoraciónProfilaxis_Adecuada/Aprobada": 1.0 if request.form["PROFILAXIS"] == "Adecuada/Aprobada" else 0.0,
                "GradoContaminación_Limpia": 1.0 if request.form["CONTAMINACION"] == "Limpia" else 0.0
            }

            # Crear vector ordenado según columnas esperadas
            datos_ordenados = [inputs[col] for col in columns]

            # Escalar e inferir
            x = scaler.transform([datos_ordenados])
            pred = model.predict(x)[0][0]
            prediction = 1 if pred >= 0.5 else 0

        except Exception as e:
            print("ERROR:", e)
            prediction = "error"

    return render_template("index.html", columns=columns, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
