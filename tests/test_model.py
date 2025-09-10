import os
import pytest
import tensorflow as tf
from src.load_model import load_model

# Ajusta el nombre del archivo a tu modelo real
MODEL_PATH = "models/WilhemNet86.h5"

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="modelo .h5 no disponible")
def test_model_loading():
    model = load_model(MODEL_PATH)
    assert isinstance(model, tf.keras.Model)
