from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import os

from predict import hacer_prediccion

app = Flask(__name__)

# Cargar modelo y scaler
model = load_model("model_mlp.h5")
scaler = joblib.load("scaler.pkl")

# Columnas que espera el modelo (sin ILQ)
columns = pd.read_csv("datos_preinfeccion_limpios.csv", sep=";", encoding="ISO-8859-1").drop("ILQ", axis=1).columns.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        form_data = request.form.to_dict()

        # ===== QUIRÓFANO (one-hot) =====
        for i in range(1, 6):
            form_data[f"Quirófano_{i}.0"] = 1 if form_data["QUIROFANO"] == str(i) else 0
        del form_data["QUIROFANO"]

        # ===== ASA (one-hot) =====
        for i in range(1, 5):
            form_data[f"Proc.ASA_{i}"] = 1 if form_data["ASA"] == str(i) else 0
        del form_data["ASA"]

        # ===== PROFILAXIS (one-hot) =====
        tipos_profilaxis = [
            "Adecuada/Aprobada",
            "Inadecuada por inicio",
            "Inadecuada por elección",
            "No administrada"
        ]
        for tipo in tipos_profilaxis:
            key = f"ValoraciónProfilaxis_{tipo}"
            form_data[key] = 1 if form_data["PROFILAXIS"] == tipo else 0
        del form_data["PROFILAXIS"]

        # ===== CONTAMINACIÓN (one-hot) =====
        grados_contaminacion = ["Limpia", "Contaminada", "Sucia"]
        for grado in grados_contaminacion:
            key = f"GradoContaminación_{grado}"
            form_data[key] = 1 if form_data["CONTAMINACION"] == grado else 0
        del form_data["CONTAMINACION"]

        # ===== Convertir valores a float y ordenar según columnas esperadas =====
        try:
            datos_usuario = [float(form_data[col]) for col in columns]
            prediction = hacer_prediccion(np.array([datos_usuario]), scaler, model)
        except Exception as e:
            print(f"Error durante la predicción: {e}")
            prediction = "error"

    return render_template("index.html", columns=columns, prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)

