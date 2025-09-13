"""Interfaz gráfica (Tkinter) para el pipeline de Neumonía + Grad-CAM.

- Carga DICOM/JPG/PNG.
- Muestra la imagen original y el heatmap.
- Ejecuta la predicción usando el pipeline modular.
- Guarda historial en CSV.
- Exporta un “pantallazo” a PDF (tkcap), nombrando como:
  reports/gui/Reporte_<CEDULA>_<YYYYMMDD-HHMMSS>.{jpg,pdf}

"""

from __future__ import annotations

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

# Silenciar logs de TensorFlow antes de importarlo indirectamente
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.data.read_img import read_dicom, read_image
from src.visualizations.integrator import run_pipeline

# Opcional para exportar PDF
try:
    import tkcap  # type: ignore
    import img2pdf  # type: ignore

    TKCAP_AVAILABLE = True
except Exception:
    TKCAP_AVAILABLE = False

# ---------------------------- Configuración -------------------------------- #

LABELS = ("bacteriana", "normal", "viral")
DEFAULT_LAST_CONV = "conv10_thisone"  
DISPLAY_SIZE = (250, 250)             # (ancho, alto) para previsualización
MODEL_PATH = "models/conv_MLP_84.h5"  # ruta por defecto del modelo .h5

OUTPUT_DIR = Path("reports/gui")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------- Utilidades ---------------------------------- #

def _np_rgb_to_pil(rgb) -> Image.Image:
    """Convierte un ndarray RGB (H, W, 3) a PIL.Image."""
    return Image.fromarray(rgb)


def _safe_id(s: str) -> str:
    """Sanitiza un ID/cédula para usarlo en nombres de archivo."""
    s = (s or "").strip()
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))


# ------------------------------ Aplicación --------------------------------- #

class App:
    """Aplicación Tkinter para detección rápida de neumonía."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Herramienta para la detección rápida de neumonía")
        self.root.geometry("820x560")
        self.root.resizable(False, False)

        bold = ("Segoe UI", 10, "bold")

        # Etiquetas
        self.lab_title = ttk.Label(
            self.root,
            text="SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA",
            font=bold,
        )
        self.lab_left = ttk.Label(self.root, text="Imagen Radiográfica", font=bold)
        self.lab_right = ttk.Label(self.root, text="Imagen con Heatmap", font=bold)
        self.lab_result = ttk.Label(self.root, text="Resultado:", font=bold)
        self.lab_prob = ttk.Label(self.root, text="Probabilidad:", font=bold)
        self.lab_id = ttk.Label(self.root, text="Cédula Paciente:", font=bold)
        self.lab_lastconv = ttk.Label(self.root, text="Última capa conv:", font=bold)

        # Entradas
        self.input_id = ttk.Entry(self.root, width=14)
        self.input_lastconv = ttk.Entry(self.root, width=18)
        self.input_lastconv.insert(0, DEFAULT_LAST_CONV)

        # Vistas de imagen
        self.img_left_box = tk.Label(
            self.root, bd=1, relief=tk.SOLID,
            width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]
        )
        self.img_right_box = tk.Label(
            self.root, bd=1, relief=tk.SOLID,
            width=DISPLAY_SIZE[0], height=DISPLAY_SIZE[1]
        )

        # Resultados
        self.text_result = tk.Text(self.root, width=12, height=1)
        self.text_prob = tk.Text(self.root, width=12, height=1)

        # Botones
        self.btn_load = ttk.Button(self.root, text="Cargar Imagen",
                                   command=self.on_load)
        self.btn_predict = ttk.Button(
            self.root, text="Predecir",
            command=self.on_predict, state=tk.DISABLED
        )
        self.btn_clear = ttk.Button(self.root, text="Borrar",
                                    command=self.on_clear)
        self.btn_save = ttk.Button(self.root, text="Guardar",
                                   command=self.on_save_csv)
        self.btn_pdf = ttk.Button(
            self.root, text="PDF",
            command=self.on_export_pdf,
            state=(tk.NORMAL if TKCAP_AVAILABLE else tk.DISABLED),
        )

        # Layout
        self.lab_title.place(x=90, y=20)
        self.lab_left.place(x=115, y=65)
        self.lab_right.place(x=555, y=65)

        self.img_left_box.place(x=65, y=90, width=DISPLAY_SIZE[0],
                                height=DISPLAY_SIZE[1])
        self.img_right_box.place(x=505, y=90, width=DISPLAY_SIZE[0],
                                 height=DISPLAY_SIZE[1])

        self.lab_id.place(x=65, y=350)
        self.input_id.place(x=200, y=350)

        self.lab_result.place(x=480, y=350)
        self.text_result.place(x=600, y=350, width=120, height=24)

        self.lab_prob.place(x=480, y=390)
        self.text_prob.place(x=600, y=390, width=120, height=24)

        self.lab_lastconv.place(x=65, y=390)
        self.input_lastconv.place(x=200, y=390)

        self.btn_load.place(x=65, y=460, width=120)
        self.btn_predict.place(x=205, y=460, width=120)
        self.btn_save.place(x=345, y=460, width=120)
        self.btn_pdf.place(x=485, y=460, width=120)
        self.btn_clear.place(x=625, y=460, width=120)

        # Estado
        self.current_rgb = None      # ndarray RGB original
        self.current_path = None     # ruta del archivo cargado
        self.tk_img_left = None      # referencias para evitar GC
        self.tk_img_right = None

        self.input_id.focus_set()
        self.root.mainloop()

    # ---------------------------- Acciones --------------------------------- #

    def on_load(self) -> None:
        """Carga una imagen DICOM/JPG/PNG, la muestra y habilita 'Predecir'."""
        filetypes = [("DICOM", "*.dcm"), ("Imagen", "*.jpg *.jpeg *.png")]
        path = filedialog.askopenfilename(
            title="Selecciona imagen",
            initialdir=str(Path.cwd()),
            filetypes=filetypes,
        )
        if not path:
            return

        try:
            if path.lower().endswith(".dcm"):
                rgb, _ = read_dicom(path)
            else:
                rgb, _ = read_image(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"No se pudo leer la imagen:\n{exc}")
            return

        self.current_rgb = rgb
        self.current_path = path

        # Mostrar imagen original
        pil = _np_rgb_to_pil(rgb).resize(DISPLAY_SIZE, Image.LANCZOS)
        self.tk_img_left = ImageTk.PhotoImage(pil)
        self.img_left_box.configure(image=self.tk_img_left)

        self.btn_predict.configure(state=tk.NORMAL)
        self._clear_results_area(right_only=True)

    def on_predict(self) -> None:
        """Ejecuta el pipeline y muestra (label, prob, heatmap)."""
        if self.current_path is None:
            messagebox.showinfo("Info", "Primero carga una imagen.")
            return

        last_conv: Optional[str] = self.input_lastconv.get().strip()
        if not last_conv:
            last_conv = DEFAULT_LAST_CONV  # fallback seguro

        try:
            label, prob, heatmap = run_pipeline(
                input_path=self.current_path,
                model_path=MODEL_PATH,  
                last_conv=last_conv,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error en predicción", str(exc))
            return

        # Mostrar resultados
        self.text_result.delete("1.0", tk.END)
        self.text_result.insert(tk.END, label)

        self.text_prob.delete("1.0", tk.END)
        self.text_prob.insert(tk.END, f"{prob:.2%}")

        pil_hm = _np_rgb_to_pil(heatmap).resize(DISPLAY_SIZE, Image.LANCZOS)
        self.tk_img_right = ImageTk.PhotoImage(pil_hm)
        self.img_right_box.configure(image=self.tk_img_right)

    def on_save_csv(self) -> None:
        """Guarda (ID, label, prob) en reports/gui/historial.csv (append)."""
        pid = self.input_id.get().strip() or "-"
        label = self.text_result.get("1.0", tk.END).strip()
        prob = self.text_prob.get("1.0", tk.END).strip()

        csv_path = OUTPUT_DIR / "historial.csv"
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";")
            writer.writerow([pid, label, prob])

        messagebox.showinfo("Guardar", f"Datos guardados en:\n{csv_path}")

    def on_export_pdf(self) -> None:
        """Exporta la ventana a PDF/JPG en reports/gui/ con cédula + timestamp."""
        if not TKCAP_AVAILABLE:
            messagebox.showwarning(
                "PDF",
                "tkcap/img2pdf no instalados. Ejecuta:\npip install tkcap img2pdf",
            )
            return

        pid = _safe_id(self.input_id.get()) or "sin_cedula"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"Reporte_{pid}_{stamp}"

        img_path = OUTPUT_DIR / f"{base}.jpg"
        pdf_path = OUTPUT_DIR / f"{base}.pdf"

        cap = tkcap.CAP(self.root)
        cap.capture(str(img_path))  # screenshot de la ventana

        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert([str(img_path)]))

        messagebox.showinfo("PDF", f"PDF generado en:\n{pdf_path}")

    def on_clear(self) -> None:
        """Limpia entradas, resultados e imágenes."""
        confirm = messagebox.askokcancel(
            "Confirmación", "Se borrarán todos los datos."
        )
        if not confirm:
            return

        self.input_id.delete(0, tk.END)
        self._clear_results_area(right_only=False)
        self.current_rgb = None
        self.current_path = None

    # ---------------------------- Utilidades -------------------------------- #

    def _clear_results_area(self, right_only: bool = False) -> None:
        """Limpia los campos/zonas de resultado."""
        self.text_result.delete("1.0", tk.END)
        self.text_prob.delete("1.0", tk.END)
        if not right_only:
            self.img_left_box.configure(image="")
            self.tk_img_left = None
        self.img_right_box.configure(image="")
        self.tk_img_right = None


def main() -> int:
    """Entrada principal de la GUI."""
    App()
    return 0


if __name__ == "__main__":
    main()
