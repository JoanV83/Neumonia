# Guía de Uso con Docker

Esta guía te ayudará a usar el detector de neumonía con Docker de manera local.

##  Inicio Rápido

### 1. Construir la imagen Docker

```bash
docker build -t neumonia-detector:latest .
```

### 2. Usar el script wrapper (Recomendado)

```bash
# Mostrar ayuda
./run_docker.sh

# Ejemplo básico
./run_docker.sh data/raw/DICOM/normal.dcm

# Con cédula de paciente
./run_docker.sh imagen.jpg --patient-id 123456789
```

### 3. Uso directo con Docker

```bash
docker run --rm \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" \
  neumonia-detector:latest \
  --input "data/raw/DICOM/normal.dcm" \
  --model "models/conv_MLP_84.h5" \
  --patient-id "123456789"
```

##  Estructura de Directorios

La imagen Docker espera esta estructura de directorios:

```
tu-proyecto/
├── models/              # Modelos .h5 (montado como volumen)
│   └── conv_MLP_84.h5
├── data/               # Imágenes de entrada (montado como volumen)
│   └── raw/DICOM/
└── reports/            # Resultados generados (montado como volumen)
    └── figures/
```

##  Comandos Útiles

### Verificar la imagen
```bash
docker images neumonia-detector
```

### Ver ayuda del programa
```bash
docker run --rm neumonia-detector:latest --help
```

### Ejecutar con docker-compose
```bash
docker-compose run --rm neumonia \
  --input "data/raw/DICOM/normal.dcm" \
  --model "models/conv_MLP_84.h5" \
  --patient-id "123456"
```

### Limpiar contenedores e imágenes
```bash
# Limpiar contenedores detenidos
docker container prune

# Limpiar imágenes no utilizadas
docker image prune

# Limpiar todo (¡Cuidado!)
docker system prune -a
```

## ⚙️ Configuración Avanzada

### Variables de Entorno Útiles

```bash
docker run --rm \
  -e TF_CPP_MIN_LOG_LEVEL=2 \     # Silenciar logs de TensorFlow
  -e PYTHONUNBUFFERED=1 \         # Output inmediato
  -v "$(pwd)/models:/app/models" \
  neumonia-detector:latest
```

### Montar directorios personalizados

```bash
docker run --rm \
  -v "/ruta/custom/modelos:/app/models" \
  -v "/ruta/custom/imagenes:/app/data" \
  -v "/ruta/custom/resultados:/app/reports" \
  neumonia-detector:latest
```

## 🔍 Solución de Problemas

### La imagen no se construye
```bash
# Verificar que Docker esté corriendo
docker info

# Limpiar cache de construcción
docker builder prune

# Construir sin cache
docker build --no-cache -t neumonia-detector:latest .
```

### Error: "No such file or directory"
- Verifica que las rutas a modelos y datos sean correctas
- Asegúrate de montar los volúmenes correctamente
- Usa rutas relativas desde el directorio del proyecto

### Advertencias de TensorFlow/CUDA
- Las advertencias sobre CUDA son normales en CPU
- Usa `TF_CPP_MIN_LOG_LEVEL=2` para silenciarlas

### El contenedor se cierra inmediatamente
```bash
# Ejecutar en modo interactivo para debug
docker run -it --rm neumonia-detector:latest /bin/bash

# Ver logs detallados
docker run --rm neumonia-detector:latest --input data/test.dcm --model models/conv_MLP_84.h5
```

##  Ejemplo Completo

Suponiendo que tienes:
- `models/conv_MLP_84.h5` (tu modelo entrenado)
- `data/raw/DICOM/paciente123.dcm` (imagen DICOM)

```bash
# 1. Construir imagen
docker build -t neumonia-detector:latest .

# 2. Ejecutar predicción
./run_docker.sh data/raw/DICOM/paciente123.dcm \
  --patient-id 123456789 \
  --model models/conv_MLP_84.h5

# 3. Ver resultados
ls -la reports/figures/
```

Los resultados aparecerán en `reports/figures/`:
- `123456789_heatmap_YYYYMMDD-HHMMSS.png`
- `123456789_resultado_YYYYMMDD-HHMMSS.txt`

## 🏥 Integración Hospitalaria

Para usar en un entorno hospitalario:

```bash
# Crear un servicio con docker-compose
docker-compose up -d neumonia

# Procesar múltiples imágenes
for imagen in data/pacientes/*.dcm; do
  ./run_docker.sh "$imagen" --patient-id "$(basename "$imagen" .dcm)"
done
```

##  Monitoreo y Logs

```bash
# Ver recursos utilizados
docker stats

# Monitorear logs en tiempo real
docker logs -f <container_id>

# Guardar logs
docker logs neumonia-detector > detector.log 2>&1
```

##  Consideraciones de Seguridad

- Los modelos `.h5` contienen información sensible
- Usa volúmenes con permisos restringidos
- No incluyas datos de pacientes en la imagen
- Considera usar secretos para credenciales

```bash
# Ejemplo con permisos restringidos
docker run --rm \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/reports:/app/reports:rw" \
  neumonia-detector:latest
```

¡Tu imagen Docker está lista para usar! 🎉
