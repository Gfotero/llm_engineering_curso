# Solución: ImportError BitNetConfig en Jupyter

## Problema

```
ImportError: cannot import name 'BitNetConfig' from 'transformers.utils.quantization_config'
```

Este error ocurre cuando el **kernel de Jupyter está usando versiones antiguas** de `transformers` o `bitsandbytes` en caché.

## ✅ Verificación

Los paquetes están correctamente instalados:
- `transformers==4.51.0` ✓
- `bitsandbytes==0.48.0` ✓
- Script Python standalone funciona correctamente ✓

## Solución: Reiniciar el Kernel de Jupyter

### Opción 1: Desde Jupyter Notebook/Lab

1. En el menú: **Kernel → Restart Kernel** (o Ctrl+R)
2. Ejecuta las celdas nuevamente desde el inicio
3. La primera celda verificará las versiones:
   ```python
   import transformers
   import bitsandbytes as bnb
   print(f"transformers: {transformers.__version__}")  # Debe mostrar 4.51.0
   print(f"bitsandbytes: {bnb.__version__}")           # Debe mostrar 0.48.0
   ```

### Opción 2: Desde VSCode

1. Click en el kernel actual (esquina superior derecha)
2. Selecciona **"Restart"**
3. Vuelve a ejecutar las celdas

### Opción 3: Forzar recarga de módulos (temporal)

Si no puedes reiniciar el kernel, añade esto al inicio del notebook:

```python
import sys
import importlib

# Limpiar módulos cacheados
if 'transformers' in sys.modules:
    del sys.modules['transformers']
if 'bitsandbytes' in sys.modules:
    del sys.modules['bitsandbytes']

# Reimportar
import transformers
import bitsandbytes
```

## Alternativa: Cargar sin cuantización

Si el error persiste después de reiniciar, usa carga en FP16:

```python
# En lugar de usar quantization_config, usa:
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    torch_dtype=torch.float16,  # FP16 en lugar de 4-bit
    token=HF_TOKEN,
    trust_remote_code=True
)
```

**Nota**: Esto usa más RAM (~16GB vs ~8GB con 4-bit) pero evita problemas con bitsandbytes.

## Verificar que funcionó

Después de reiniciar, la celda de verificación debe mostrar:

```
transformers: 4.51.0
bitsandbytes: 0.48.0

⚠️ Si hay error de ImportError, reinicia el kernel de Jupyter
```

Si ves versiones diferentes, el kernel sigue usando el entorno viejo.

## Test sin Jupyter

Para verificar que tu configuración funciona, ejecuta:

```bash
python test_model_load_minimal.py
```

Este script prueba la configuración fuera de Jupyter y debe pasar todos los checks.
