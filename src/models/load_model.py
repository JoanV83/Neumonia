"""Carga y validación de modelos Keras (.h5)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import tensorflow as tf


def load_model(model_path: Union[str, Path]) -> tf.keras.Model:
    """Carga un modelo Keras desde disco y valida su tipo.

    Parameters
    ----------
    model_path : str | Path
        Ruta al archivo `.h5`.

    Returns
    -------
    tf.keras.Model
        Modelo Keras listo para inferencia.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo del modelo.
    TypeError
        Si lo cargado no es un `tf.keras.Model`.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el modelo: {p}")

    model = tf.keras.models.load_model(str(p), compile=False)
    if not isinstance(model, tf.keras.Model):
        raise TypeError("El archivo cargado no es un tf.keras.Model")
    return model
