#!/bin/bash
# Script para ejecutar el detector de neumonía en Docker

# Configuración por defecto
IMAGEN="neumonia-detector:latest"
MODELO_DEFAULT="models/conv_MLP_84.h5"
LAST_CONV_DEFAULT="conv10_thisone"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar ayuda
show_help() {
    echo -e "${GREEN}🫁 Detector de Neumonía - Docker Wrapper${NC}"
    echo -e "${YELLOW}Uso:${NC}"
    echo "  $0 <imagen.dcm|jpg|png> [opciones]"
    echo ""
    echo -e "${YELLOW}Opciones:${NC}"
    echo "  --model PATH          Ruta al modelo (default: ${MODELO_DEFAULT})"
    echo "  --last-conv LAYER     Capa convolucional (default: ${LAST_CONV_DEFAULT})"
    echo "  --patient-id ID       Cédula del paciente"
    echo "  --help               Mostrar esta ayuda"
    echo ""
    echo -e "${YELLOW}Ejemplos:${NC}"
    echo "  $0 data/raw/DICOM/normal.dcm"
    echo "  $0 imagen.jpg --patient-id 123456789"
    echo "  $0 radiografia.png --model models/mi_modelo.h5"
    echo ""
    echo -e "${YELLOW}Estructura de directorios esperada:${NC}"
    echo "  ./models/    - Modelos .h5"
    echo "  ./data/      - Imágenes de entrada"
    echo "  ./reports/   - Resultados de salida"
}

# Verificar si Docker está corriendo
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker no está corriendo${NC}"
    echo "Inicia Docker Desktop e intenta nuevamente"
    exit 1
fi

# Verificar si existe la imagen
if ! docker image inspect $IMAGEN >/dev/null 2>&1; then
    echo -e "${RED}❌ Error: La imagen $IMAGEN no existe${NC}"
    echo "Construye la imagen primero con:"
    echo -e "${BLUE}docker build -t $IMAGEN .${NC}"
    exit 1
fi

# Si no hay argumentos, mostrar ayuda
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

# Procesar argumentos
INPUT=""
MODEL="$MODELO_DEFAULT"
LAST_CONV="$LAST_CONV_DEFAULT"
PATIENT_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --last-conv)
            LAST_CONV="$2"
            shift 2
            ;;
        --patient-id)
            PATIENT_ID="$2"
            shift 2
            ;;
        *)
            if [ -z "$INPUT" ]; then
                INPUT="$1"
            else
                echo -e "${RED}❌ Error: Argumento inesperado: $1${NC}"
                exit 1
            fi
            shift
            ;;
    esac
done

# Verificar que se especificó una imagen
if [ -z "$INPUT" ]; then
    echo -e "${RED}❌ Error: Debes especificar una imagen de entrada${NC}"
    show_help
    exit 1
fi

# Verificar que existe la imagen de entrada
if [ ! -f "$INPUT" ]; then
    echo -e "${RED}❌ Error: El archivo $INPUT no existe${NC}"
    exit 1
fi

# Verificar que existe el modelo
if [ ! -f "$MODEL" ]; then
    echo -e "${RED}❌ Error: El modelo $MODEL no existe${NC}"
    echo "Asegúrate de que el archivo .h5 esté en la ubicación correcta"
    exit 1
fi

# Crear directorios si no existen
mkdir -p reports/figures

# Construir comando Docker
CMD_ARGS="--input $INPUT --model $MODEL --last-conv $LAST_CONV"
if [ ! -z "$PATIENT_ID" ]; then
    CMD_ARGS="$CMD_ARGS --patient-id $PATIENT_ID"
fi

# Mostrar información de ejecución
echo -e "${GREEN}🚀 Ejecutando detector de neumonía...${NC}"
echo -e "${BLUE}📁 Entrada:${NC} $INPUT"
echo -e "${BLUE}🧠 Modelo:${NC} $MODEL"
echo -e "${BLUE}🔧 Capa conv:${NC} $LAST_CONV"
if [ ! -z "$PATIENT_ID" ]; then
    echo -e "${BLUE}👤 Paciente:${NC} $PATIENT_ID"
fi
echo ""

# Ejecutar Docker
docker run --rm \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/reports:/app/reports" \
    -e TF_CPP_MIN_LOG_LEVEL=2 \
    $IMAGEN $CMD_ARGS

# Verificar resultado
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Proceso completado exitosamente${NC}"
    echo -e "${YELLOW}📊 Los resultados están en:${NC} reports/figures/"
    ls -la reports/figures/ | tail -3
else
    echo -e "${RED}❌ Error durante la ejecución${NC}"
    exit 1
fi
