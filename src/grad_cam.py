"""Generación de Grad-CAM.

Responsabilidades:
- Integración con el modelo.
- Devolver clase, probabilidad y heatmap.
"""

from typing import Tuple
import numpy as np
import tensorflow as tf
import cv2


def grad_cam(model: tf.keras.Model,
             x: np.ndarray,
             last_conv_name: str,
             base_rgb: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Calcula Grad-CAM y devuelve clase, prob. y heatmap colorizado.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo Keras ya cargado.
    x : np.ndarray
        Tensor de entrada con forma (1, 512, 512, 1).
    last_conv_name : str
        Nombre de la última capa convolucional del modelo.
    base_rgb : np.ndarray
        Imagen base RGB (H, W, 3) para superponer el heatmap.

    Returns
    -------
    tuple
        (class_idx, probability, heatmap_rgb) donde `heatmap_rgb` es
        un np.ndarray (512, 512, 3) uint8 con el mapa de calor
        superpuesto a la imagen base.
    """
    # Predicción
    preds = model(x, training=False).numpy()[0]
    class_idx = int(np.argmax(preds))
    prob = float(preds[class_idx])

    # Modelo intermedio para obtener activaciones de la capa conv
    conv_layer = model.get_layer(last_conv_name)
    cam_model = tf.keras.Model([model.inputs],
                               [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, logits = cam_model(x, training=False)
        class_channel = logits[:, class_idx]

    grads = tape.gradient(class_channel, conv_out)  # (1, Hc, Wc, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    conv_out = conv_out[0] * pooled_grads  # ponderación por canal
    heatmap = tf.reduce_mean(conv_out, axis=-1).numpy()  # (Hc, Wc)

    # ReLU y normalización
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    # Redimensionar al tamaño destino (512×512) y colorizar
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

    # Superponer sobre la imagen base redimensionada
    base_resized = cv2.resize(base_rgb, (512, 512))
    overlay = cv2.addWeighted(base_resized, 0.2, heatmap_color, 0.8, 0)

    return class_idx, prob, overlay
