import numpy as np
from src.preprocess_img import preprocess_image

def test_preprocess_output_shape():
    # imagen dummy (H, W, 3) en RGB
    dummy = (np.random.rand(720, 960, 3) * 255).astype(np.uint8)
    x = preprocess_image(dummy)
    assert x.shape == (1, 512, 512, 1)
    assert x.dtype == np.float32
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0
