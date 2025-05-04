from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
from predict import hacer_prediccion
import os  # <--- importante para leer el puerto de Render

app = Flask(__name__)

# Cargar modelo y scaler
model = load_model("model_mlp.h5")
scaler = joblib.load("scaler.pkl")

# Columnas esperadas
columns = pd.read_csv("datos_preinfeccion_limpios.csv", sep=";", encoding="ISO-8859-1").drop("ILQ", axis=1).columns.tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        datos_usuario = [float(request.form[col]) for col in columns]
        prediction = hacer_prediccion(np.array([datos_usuario]), scaler, model)
    return render_template("index.html", columns=columns, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

