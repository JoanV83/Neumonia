"""Smoke test del pipeline (usa cédula + timestamp al guardar)."""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # menos ruido de TensorFlow

from src.visualizations.integrator import run_pipeline, save_outputs


def main() -> None:
    """Ejecuta una predicción de prueba y guarda los artefactos."""
    input_path = "data/raw/DICOM/normal (2).dcm"
    model_path = "models/conv_MLP_84.h5"
    last_conv = "conv10_thisone"        
    patient_id = "1234567890"           # ← cambiar por la cédula deseada

    # 1) Pipeline
    label, prob, heatmap = run_pipeline(
        input_path=input_path,
        model_path=model_path,
        last_conv=last_conv,
    )

    # 2) Guardar con cédula + timestamp en reports/figures/
    heatmap_path, txt_path = save_outputs(
        outdir="reports/figures",
        label=label,
        prob=prob,
        heatmap_rgb=heatmap,
        patient_id=patient_id,
    )

    # 3) Mensaje final
    print(f"Predicción: {label} | Prob: {prob:.4f} | Heatmap shape: {heatmap.shape}")
    print("Guardado en:")
    print(f"  {heatmap_path}")
    print(f"  {txt_path}")


if __name__ == "__main__":
    main()