"""Carga y validación de modelos Keras (.h5).

Responsabilidades:
- Cargar `WilhemNet86.h5` (u otros .h5).
- Validar integridad del modelo.
"""

import os
import tensorflow as tf


def load_model(model_path: str) -> tf.keras.Model:
    """Carga un modelo Keras desde disco y valida su tipo.

    Parameters
    ----------
    model_path : str
        Ruta al archivo .h5.

    Returns
    -------
    tf.keras.Model
        Modelo Keras listo para inferencia.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo del modelo.
    TypeError
        Si lo cargado no es un tf.keras.Model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el modelo: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    if not isinstance(model, tf.keras.Model):
        raise TypeError("El archivo cargado no es un tf.keras.Model")
    return model
