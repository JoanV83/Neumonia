import pytest

from src.data.read_img import read_image

def test_read_image_raises_not_found(tmp_path):
    missing = tmp_path / "no_existe.jpg"
    with pytest.raises(FileNotFoundError):
        read_image(str(missing))