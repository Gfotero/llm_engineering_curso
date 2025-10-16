# ✅ Resumen: Persistencia Offline de Llama 3.1

## Estado: COMPLETADO

### Archivos Creados

1. **Scripts principales**:
   - `save_model_offline.py` - Guarda tokenizer y descarga snapshot
   - `load_model_offline.py` - Carga modelo offline con cuantización

2. **Documentación**:
   - `OFFLINE_USAGE.md` - Guía completa de uso offline
   - `PERSISTENCIA_RESUMEN.md` - Este resumen

3. **Notebook actualizado**:
   - `week3/day4.ipynb` - Añadidas celdas de persistencia y carga offline

4. **Configuración**:
   - `.gitignore` - Actualizado con `artifacts/` y `hf_mirror/`

## Estructura de Directorios

```
proyecto/
├── save_model_offline.py          # Script para guardar
├── load_model_offline.py          # Script para cargar offline
├── OFFLINE_USAGE.md               # Documentación completa
│
├── artifacts/llama31-8b/          # Tokenizer guardado (ignorado por git)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── hf_mirror/llama31-8b/          # Snapshot local (opcional, ignorado por git)
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── model-*.safetensors (~16GB)
│
└── ~/.cache/huggingface/hub/      # Caché automático (recomendado)
    └── models--meta-llama--Meta-Llama-3.1-8B-Instruct/
```

## ⚠️ Limitación IMPORTANTE

**Los modelos cuantizados con BitsAndBytes NO se pueden guardar con `save_pretrained()`**

### ¿Por qué?

- BitsAndBytes cuantiza en **memoria** durante la carga
- Los pesos 4-bit están en formato interno de `bitsandbytes`
- `save_pretrained()` solo soporta formatos estándar (FP32/FP16/BF16)

### Solución Implementada

1. ✓ Guardar modelo **original** (FP16) en caché
2. ✓ Re-aplicar cuantización **cada vez** que se carga
3. ✓ La cuantización es rápida (~1-2 min) vs descargar 16GB

## Flujo de Uso

### Primera vez (ONLINE)

```bash
# 1. Guardar modelo y tokenizer
python save_model_offline.py

# Esto creará:
# - ./artifacts/llama31-8b/ (tokenizer)
# - ./hf_mirror/llama31-8b/ (modelo completo ~16GB)
# - ~/.cache/huggingface/ (caché automático)
```

### Uso posterior (OFFLINE)

```bash
# 2. Cargar offline
python load_model_offline.py

# O en modo offline estricto:
TRANSFORMERS_OFFLINE=1 python load_model_offline.py
```

### En Jupyter Notebook

```python
# Ver week3/day4.ipynb, sección "Persistencia para Uso Offline"

# 1. Guardar tokenizer
from pathlib import Path
ARTIFACTS_DIR = Path("./artifacts/llama31-8b")
tokenizer.save_pretrained(ARTIFACTS_DIR)

# 2. Cargar offline
tokenizer_offline = AutoTokenizer.from_pretrained(
    str(ARTIFACTS_DIR),
    local_files_only=True
)

# 3. Cargar modelo con re-cuantización
model_offline = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config,
    local_files_only=True
)
```

## Variables de Entorno

### HF_HOME
Directorio base de caché de HuggingFace:
```bash
export HF_HOME=/path/to/custom/cache
# Default: ~/.cache/huggingface
```

### TRANSFORMERS_CACHE
Caché específico de transformers:
```bash
export TRANSFORMERS_CACHE=/path/to/cache
# Default: $HF_HOME/hub
```

### Modo Offline Estricto
```bash
# Opción 1 (transformers)
export TRANSFORMERS_OFFLINE=1

# Opción 2 (huggingface_hub)
export HF_HUB_OFFLINE=1
```

## Prioridad de Carga

El sistema intenta cargar en este orden:

1. **Mirror local** (`./hf_mirror/llama31-8b/`)
2. **Caché de HF** (`~/.cache/huggingface/hub/`) ← **Más común**
3. **Descarga online** (solo si `local_files_only=False`)

## Espacio en Disco

- Modelo original (FP16): **~16 GB** (disco)
- Modelo cuantizado (4-bit): **~4-5 GB** (RAM)
- Tokenizer: **~5 MB**

## Performance

| Operación | Tiempo |
|-----------|--------|
| Primera carga (online) | ~5-10 min (descarga 16GB) |
| Carga offline + cuantización | ~1-2 min |
| Solo tokenizer | <1 seg |

## Comandos Útiles

### Verificar caché
```bash
# Ver modelos cacheados
ls -lh ~/.cache/huggingface/hub/

# Tamaño del caché
du -sh ~/.cache/huggingface/hub/

# Modelo Llama específico
ls -lh ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/
```

### Probar carga offline
```bash
# Test completo
python load_model_offline.py

# Modo offline estricto (sin red)
TRANSFORMERS_OFFLINE=1 python load_model_offline.py
```

### Jupyter
```bash
# Ver notebook con ejemplos
jupyter lab week3/day4.ipynb
```

## Troubleshooting

### "Repository not found" en modo offline

**Solución**: Ejecuta primero `python save_model_offline.py` online

### "Cannot save quantized model"

**Esto es normal**: Los modelos 4-bit no se guardan, se re-cuantizan al cargar

### Carga lenta

**Verificar**: Puede estar descargando de internet
```bash
# Monitorear tráfico
nethogs

# Forzar offline
TRANSFORMERS_OFFLINE=1 python load_model_offline.py
```

## Referencias Rápidas

- **Guía completa**: `OFFLINE_USAGE.md`
- **Script de guardado**: `save_model_offline.py`
- **Script de carga**: `load_model_offline.py`
- **Notebook**: `week3/day4.ipynb`
- **Docs HF**: https://huggingface.co/docs/transformers/installation#offline-mode

## Próximos Pasos

1. ✅ Ejecuta `python save_model_offline.py` (si no lo has hecho)
2. ✅ Prueba `python load_model_offline.py`
3. ✅ Experimenta con el notebook `week3/day4.ipynb`
4. ✅ En producción, considera usar `HF_HOME` personalizado

---

**✅ Sistema de persistencia offline completado y documentado**
