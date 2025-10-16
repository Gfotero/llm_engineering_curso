# Configuración de Hugging Face

## Variables de entorno

Este proyecto usa `HF_TOKEN` como variable estándar para autenticación con Hugging Face.

### Configuración del .env

1. **No commitear el archivo .env** - Ya está incluido en `.gitignore`
2. Añadir tu token en `.env`:
   ```
   HF_TOKEN=tu_token_aqui
   ```

### Obtener un token de Hugging Face

1. Ve a https://huggingface.co/settings/tokens
2. Crea un nuevo token con permisos de lectura
3. Para modelos con licencia (como Llama), acepta la licencia del modelo en su página

## Verificación

### Test 1: Validación de autenticación básica

```bash
python hf_check.py
```

Este script verifica:
- ✓ Que `HF_TOKEN` esté definido
- ✓ Que la autenticación funcione
- ✓ Que puedas descargar tokenizers de modelos con licencia

### Test 2: Acceso al modelo (sin descargar pesos)

```bash
python test_model_access.py
```

Valida que puedas acceder al config y tokenizer del modelo sin descargar los pesos completos (~16GB).

### Test 3: Cargar modelo completo (requiere GPU/RAM)

Usa el notebook `week3/day4.ipynb` que incluye:
- Login automático con `HF_TOKEN`
- Carga del modelo con cuantización 4-bit
- Ejemplo de generación de texto con streaming

**Requisitos**: GPU con CUDA o ~16GB RAM mínimo

## Solución de problemas

### Error 401/GatedRepoError

Si obtienes este error:
1. Verifica que aceptaste la licencia del modelo en HuggingFace
2. Confirma que tu token tiene permisos de lectura
3. Ejecuta `python hf_check.py` para diagnosticar

### Usar CLI de Hugging Face

```bash
# Login (método actualizado)
huggingface-cli login

# Verificar sesión
hf auth whoami
```

## Versiones recomendadas

- `transformers==4.51.0` (estable, evita issues de propagación de token)
- `huggingface-hub>=0.20.0`

## Migración desde versiones anteriores

Si usabas `HUGGINGFACE_HUB_TOKEN`, actualiza tu código a `HF_TOKEN`:

```python
# Antes
HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Ahora
HF_TOKEN = os.getenv("HF_TOKEN")
```

## Modelo alternativo para testing

Si Llama 3.1 da problemas, prueba con un modelo abierto:

```python
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
```
