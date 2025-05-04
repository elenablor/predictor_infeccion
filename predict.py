
import numpy as np

def hacer_prediccion(X, scaler, model, umbral=0.3):
    X_scaled = scaler.transform(X)
    proba = model.predict(X_scaled).flatten()[0]
    return int(proba > umbral)
