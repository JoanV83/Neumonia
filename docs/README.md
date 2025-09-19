# Herramienta para la detección rápida de neumonía

Clasificación de radiografías de tórax (DICOM/JPG/PNG) en tres categorías mediante Deep Learning:

1) **Neumonía bacteriana**  
2) **Neumonía viral**  
3) **Sin neumonía**

El sistema genera explicaciones con **Grad-CAM**, superponiendo un mapa de calor sobre la imagen para resaltar las regiones más relevantes.

---

## 🚦 Estado del proyecto

- **Python recomendado:** **3.11.4** (probado con TensorFlow 2.18 + Keras 3.8)  
- **Sistema:** Windows / Linux / macOS (para GUI en contenedor no aplica)  
- **Modelo por defecto:** `models/conv_MLP_84.h5`



---

## 🖼️ Vistas & ejemplos

- **GUI (Tkinter)**  
  `docs/img/ui_tk.png`

- **Grad-CAM (CLI/Smoke)**  
  `docs/img/gradcam_example.png`


---

## ✨ Características

- Lectura de **DICOM/JPG/PNG** → normalización a **RGB**.
- Preprocesamiento: **grises**, **resize 512×512**, **CLAHE**, **[0,1]**, tensor **(1,H,W,1)**.
- Inferencia con modelo Keras/TensorFlow (.h5).
- **Grad-CAM** (overlay RGB) explicable.
- **CLI** y **GUI** (Tkinter).
- Reportes nombrados con **cédula + timestamp**.
- Suite de pruebas con **pytest**.

---

## 📦 Instalación

### Opción A — venv + pip (recomendada)

#### PowerShell (Windows)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

#### Bash (Linux/macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


### Opción B — Conda (opcional)
```bash
conda create -n tf python=3.10 -y
conda activate tf
pip install -r requirements.txt
```

---

## 🚀 Uso rápido

### 1) Smoke test (validación mínima end-to-end)
```powershell
python -m scripts.smoke
```
- Imprime *label* + probabilidad.
- Guarda: `reports/figures/<CEDULA>_heatmap_YYYYMMDD-HHMMSS.png` + `.txt`  
  (La cédula se define en `scripts/smoke.py`).

### 2) CLI (pipeline completo)
```powershell
python -m src.visualizations.integrator `
  --input "data/raw/DICOM/normal (2).dcm" `
  --model "models/conv_MLP_84.h5" `
  --last-conv "conv10_thisone" `
  --patient-id "123456789" `
  --outdir "reports/figures"
```

### 3) GUI (Tkinter)
```powershell
python -m src.visualizations.ui_tk
```
Flujo en interfaz:
- Ingrese **cédula**.
- **Cargar Imagen** → seleccione DICOM/JPG/PNG.
- **Predecir** → muestra clase, prob y Grad-CAM.
- **Guardar** → `reports/gui/historial.csv`.
- **PDF** → `reports/gui/Reporte_<CEDULA>_<YYYYMMDD-HHMMSS>.pdf` (+ .jpg).


---

## 🐳 Docker



### Dockerfile 
```dockerfile
# Imagen base ligera para CPU
FROM python:3.11-slim

# Configurar entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema necesarias para OpenCV-headless y TensorFlow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements.txt y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código fuente completo
COPY src/ ./src/
COPY scripts/ ./scripts/

# Crear directorios para modelos, datos y reportes
RUN mkdir -p models data reports

# Configurar entrada usando el módulo correcto
ENTRYPOINT ["python", "-m", "src.visualizations.integrator"]
CMD ["--help"]
```

```bash
docker build -t neumonia:cli .
```


```bash
# Bash (Linux/macOS)
docker run --rm -it \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/reports:/app/reports" \
  neumonia:cli \
  --input "data/raw/DICOM/normal (2).dcm" \
  --model "models/conv_MLP_84.h5" \
  --last-conv "conv10_thisone" \
  --outdir "reports/figures"
```

```powershell
# Powershell (Windows)
docker run --rm -it `
  -v "$PWD/data:/app/data" `
  -v "$PWD/models:/app/models" `
  -v "$PWD/reports:/app/reports" `
  neumonia:cli `
  --input "data/raw/DICOM/normal (2).dcm" `
  --model "models/conv_MLP_84.h5" `
  --last-conv "conv10_thisone" `
  --outdir "reports/figures"
```

---

## 🗂️ Estructura del proyecto

```
.
├── data/
│   └── raw/...
├── models/
│   └── conv_MLP_84.h5
├── reports/
│   ├── figures/      
│   └── gui/          
├── scripts/
│   └── smoke.py
├── src/
│   ├── data/
│   │   ├── read_img.py
│   │   └── preprocess_img.py
│   ├── models/
│   │   ├── load_model.py
│   │   └── grad_cam.py
│   └── visualizations/
│       ├── integrator.py
│       └── ui_tk.py
├── tests/
│   └── test_*.py
├── docs/
│   └── img/ 
|   └── README.md   
├── requirements.txt
├── gitignore
├── LICENCE.txt
└── Dockerfile
```

---

## 🧪 Pruebas

Ejecutar toda la suite:
```bash
pytest -q
```

Comandos útiles:
```bash
pytest -vv              # detallado
pytest -k preprocess    # por patrón
pytest --last-failed    # solo fallos previos
pytest --durations=5    # tests más lentos
```

---

## 🔧 Troubleshooting

- **PowerShell: “la ejecución de scripts está deshabilitada”**  
  Abre PowerShell **como admin** y ejecuta:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```


- **DICOM comprimidos no abren**  
  Asegúrate de tener: `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`.

---

## 🧠 Detalles técnicos 

- **Preprocesamiento:** gris → 512×512 → CLAHE → [0,1] → (1,512,512,1).
- **Modelo:** Keras/TensorFlow (`.h5`)  
  Última capa conv por defecto: `conv10_thisone` (configurable con `--last-conv`).
- **Grad-CAM:** Gradientes dirigidos a la clase para ponderar activaciones de la última capa conv y superponer en RGB.

---

## 📜 Licencia

Se distribuye con licencia **MIT**.
```
MIT License

Copyright (c) 2025 ...
```
Texto completo: https://opensource.org/license/mit/

---

## 👩🏽‍💻 Créditos

Proyecto original:
- **Isabella Torres Revelo** — https://github.com/isa-tr
- **Nicolás Díaz Salazar** — https://github.com/nicolasdiazsalazar

Proyecto adaptado y modificado:
- **Joan Andres Velasquez** — https://github.com/JoanV83
- **Edwin Vicente Zapata** — https://github.com/edwinviz
- **Miguel Saavedra** — https://github.com/mash4403
- **Andres Velasco** — https://github.com/Andres-Velasco07


