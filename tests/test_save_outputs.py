from pathlib import Path
import re
import numpy as np
from src.visualizations.integrator import save_outputs

def test_save_outputs_with_patient_id(tmp_path):
    # heatmap dummy (RGB 512x512)
    heatmap = (np.random.rand(512, 512, 3) * 255).astype("uint8")
    outdir = tmp_path / "reports" / "figures"
    outdir_str = str(outdir)

    heatmap_path, txt_path = save_outputs(
        outdir=outdir_str,
        label="normal",
        prob=0.99,
        heatmap_rgb=heatmap,
        patient_id="123-ABC_45",   # se saneará a 123-ABC_45
    )

    # existen
    assert Path(heatmap_path).exists()
    assert Path(txt_path).exists()

    # nombre: 123-ABC_45_heatmap_YYYYMMDD-HHMMSS.png
    assert re.search(r"123-ABC_45_heatmap_\d{8}-\d{6}\.png$", heatmap_path)
    assert re.search(r"123-ABC_45_resultado_\d{8}-\d{6}\.txt$", txt_path)

    # contenido mínimo
    with open(txt_path, "r", encoding="utf-8") as f:
        s = f.read()
    assert "clase=normal" in s