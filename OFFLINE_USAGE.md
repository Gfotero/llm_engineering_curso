# Uso Offline de Llama 3.1 con Cuantización 4-bit

## Resumen

Esta guía documenta cómo persistir y reusar el modelo Llama 3.1 offline, aplicando cuantización 4-bit en cada carga.

## ⚠️ Limitación Importante: Modelos Cuantizados

**Los modelos cuantizados con BitsAndBytes NO se pueden guardar directamente con `save_pretrained()`.**

### ¿Por qué?

- `BitsAndBytes` cuantiza el modelo **en memoria** durante la carga
- Los pesos cuantizados están en formato interno de `bitsandbytes`, no en formato estándar de PyTorch
- `save_pretrained()` solo guarda modelos en formatos estándar (FP32, FP16, BF16)

### Solución

1. **Guardar el modelo original** (FP16/BF16) en caché local
2. **Re-aplicar cuantización** cada vez que se carga
3. La cuantización es rápida (~1-2 min) comparado con descargar 16GB de internet

## Arquitectura de Persistencia

```
proyecto/
├── artifacts/llama31-8b/          # Tokenizer guardado
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── hf_mirror/llama31-8b/           # Snapshot local completo (opcional)
│   ├── config.json
│   ├── model.safetensors.index.json
│   ├── model-00001-of-00004.safetensors
│   └── ...
│
└── ~/.cache/huggingface/hub/       # Caché automático de HF (recomendado)
    └── models--meta-llama--Meta-Llama-3.1-8B-Instruct/
        └── snapshots/<revision>/
            └── [archivos del modelo]
```

## Variables de Entorno

### HF_HOME
Directorio base para todos los archivos de Hugging Face:
```bash
export HF_HOME=/path/to/custom/cache
```
Default: `~/.cache/huggingface`

### TRANSFORMERS_CACHE
Caché específico de transformers:
```bash
export TRANSFORMERS_CACHE=/path/to/transformers/cache
```
Default: `$HF_HOME/hub`

### Modo Offline Estricto

Fuerza a transformers/huggingface_hub a NO conectarse a internet:

```bash
# Opción 1: transformers
export TRANSFORMERS_OFFLINE=1

# Opción 2: huggingface_hub
export HF_HUB_OFFLINE=1

# Cargar modelo
python load_model_offline.py
```

## Flujo de Trabajo

### 1. Primera vez: Descargar y guardar (ONLINE)

```bash
# Ejecutar con conexión a internet
python save_model_offline.py
```

Este script:
- ✓ Guarda tokenizer en `./artifacts/llama31-8b/`
- ✓ Descarga snapshot completo a `./hf_mirror/llama31-8b/` (~16GB)
- ✓ El modelo también se cachea automáticamente en `~/.cache/huggingface/`

### 2. Uso posterior: Cargar offline (OFFLINE)

```bash
# Opción A: Modo offline estricto
TRANSFORMERS_OFFLINE=1 python load_model_offline.py

# Opción B: Permitir conexión fallback
python load_model_offline.py
```

Este script:
1. Carga tokenizer desde `./artifacts/` (o caché)
2. Carga modelo desde `./hf_mirror/` o `~/.cache/huggingface/`
3. **Aplica cuantización 4-bit AL CARGAR**
4. Ejecuta inferencia de prueba

## Prioridad de Carga

El script intenta cargar en este orden:

1. **Mirror local**: `./hf_mirror/llama31-8b/` (si existe)
2. **Caché de HF**: `~/.cache/huggingface/hub/` (más común)
3. **Descarga online**: Solo si `local_files_only=False` (default)

## Código de Ejemplo

### Guardar para uso offline

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ARTIFACTS_DIR = Path("./artifacts/llama31-8b")

# Guardar tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=HF_TOKEN)
tokenizer.save_pretrained(ARTIFACTS_DIR)

# Descargar snapshot completo
snapshot_download(
    repo_id=LLAMA,
    token=HF_TOKEN,
    local_dir="./hf_mirror/llama31-8b",
    local_dir_use_symlinks=False
)
```

### Cargar offline con cuantización

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Configurar cuantización (se aplica al cargar)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Cargar tokenizer offline
tokenizer = AutoTokenizer.from_pretrained(
    "./artifacts/llama31-8b",
    local_files_only=True
)

# Cargar modelo offline con cuantización
model = AutoModelForCausalLM.from_pretrained(
    "./hf_mirror/llama31-8b",  # o usar nombre del modelo si está cacheado
    device_map="auto",
    quantization_config=quant_config,
    local_files_only=True,
    trust_remote_code=True
)
```

## Verificar Caché Existente

```bash
# Ver modelos cacheados
ls -lh ~/.cache/huggingface/hub/

# Ver modelo Llama específico
ls -lh ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/

# Tamaño del caché
du -sh ~/.cache/huggingface/hub/
```

## Mover Caché a Otro Directorio

```bash
# Copiar caché existente
cp -r ~/.cache/huggingface /mnt/external/hf_cache

# Configurar HF_HOME
export HF_HOME=/mnt/external/hf_cache

# Verificar
python -c "from huggingface_hub import constants; print(constants.HF_HOME)"
```

## Troubleshooting

### Error: "Repository not found" en modo offline

**Causa**: El modelo no está en caché local.

**Solución**:
```bash
# Descargar primero online
python save_model_offline.py

# Luego usar offline
TRANSFORMERS_OFFLINE=1 python load_model_offline.py
```

### Error: "Cannot save quantized model"

**Causa**: Intentando hacer `save_pretrained()` de modelo cuantizado.

**Solución**: Esta es una limitación conocida. Los modelos 4-bit **no** se pueden guardar, solo re-cuantizar al cargar.

### Modelo carga muy lento

**Causa**: Está descargando desde internet.

**Verificar**:
```bash
# Monitorear tráfico de red
sudo nethogs  # o: iftop

# Verificar que use caché
strace -e open python load_model_offline.py 2>&1 | grep cache
```

## Espacio en Disco

- **Modelo original (FP16)**: ~16 GB
- **Modelo cuantizado (4-bit) en RAM**: ~4-5 GB
- **Tokenizer**: ~5 MB
- **Total en disco**: ~16 GB
- **Total en RAM durante uso**: ~5 GB

## Performance

- **Primera carga (online)**: ~5-10 min (descarga 16GB)
- **Carga offline + cuantización**: ~1-2 min
- **Carga solo tokenizer**: <1 seg

## Referencias

- [Transformers Offline Mode](https://huggingface.co/docs/transformers/installation#offline-mode)
- [HuggingFace Hub Cache](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
- [BitsAndBytes Quantization](https://huggingface.co/docs/transformers/main_classes/quantization)
