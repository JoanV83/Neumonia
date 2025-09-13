import numpy as np

from src.data.preprocess_img import preprocess_image


def test_preprocess_output_shape():
    """Confirma forma, tipo y rango del tensor preprocesado."""
    dummy = (np.random.rand(720, 960, 3) * 255).astype(np.uint8)
    x = preprocess_image(dummy)
    assert x.shape == (1, 512, 512, 1)
    assert x.dtype == np.float32
    assert 0.0 <= float(x.min()) <= 1.0
    assert 0.0 <= float(x.max()) <= 1.0