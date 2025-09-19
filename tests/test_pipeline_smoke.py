import numpy as np

from src.data.preprocess_img import preprocess_image


def test_pipeline_smoke():
    """Asegura que el preprocesamiento no rompe con ruido."""
    dummy = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    x = preprocess_image(dummy)
    assert x.ndim == 4 and x.shape[-1] == 1
