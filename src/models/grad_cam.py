"""Generación de Grad-CAM y superposición con la imagen base (RGB).

En memoria se trabaja con **RGB**; solo se convierte a BGR al guardar.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf


def _ensure_layer_exists(model: tf.keras.Model, layer_name: str) -> None:
    try:
        model.get_layer(layer_name)
    except Exception as exc:  # noqa: BLE001
        layer_names = [l.name for l in model.layers]
        raise ValueError(
            f"No existe la capa '{layer_name}'. Capas disponibles: {layer_names}"
        ) from exc


def grad_cam(
    model: tf.keras.Model,
    x: np.ndarray,
    last_conv_name: str,
    base_rgb: np.ndarray,
) -> Tuple[int, float, np.ndarray]:
    """Calcula Grad-CAM y devuelve (clase, probabilidad, heatmap superpuesto RGB).

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras ya cargado.
    x : np.ndarray
        Tensor de entrada con forma (1, 512, 512, 1).
    last_conv_name : str
        Nombre de la última capa convolucional.
    base_rgb : np.ndarray
        Imagen base RGB (H, W, 3) para superponer el heatmap.

    Returns
    -------
    tuple
        (class_idx, probability, heatmap_rgb_uint8)
    """
    _ensure_layer_exists(model, last_conv_name)

    # Predicción para clase y probabilidad
    preds = model.predict(x, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    prob = float(preds[class_idx])

    # Modelo intermedio para activaciones de la capa conv
    conv_layer = model.get_layer(last_conv_name)
    cam_model = tf.keras.Model(inputs=model.input,
                            outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, logits = cam_model(x, training=False)
        class_channel = logits[:, class_idx]

    grads = tape.gradient(class_channel, conv_out)  # (1, Hc, Wc, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_out[0] * pooled_grads  # ponderación canal a canal
    heatmap = tf.reduce_mean(conv_out, axis=-1).numpy()  # (Hc, Wc)

    # ReLU + normalización
    heatmap = np.maximum(heatmap, 0.0)
    heatmap /= (heatmap.max() + 1e-8)

    # Redimensionar al tamaño destino y colorizar (mantener RGB)
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Superposición
    base_resized = cv2.resize(base_rgb, (512, 512))
    overlay = cv2.addWeighted(base_resized, 0.3, heatmap_rgb, 0.7, 0.0)

    return class_idx, prob, overlay
