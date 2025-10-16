# ✅ Cómo Usar Llama 3.1 - Guía Rápida

## Estado Actual

✅ **Modelo guardado y funcionando**
- Ubicación: `~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/`
- Tamaño: 15 GB (modelo completo en FP16)
- Memoria al cargar con cuantización: ~5.6 GB

## Formas de Usar el Modelo

### 1. Script Python Simple (Recomendado para empezar)

```bash
python demo_llama_simple.py
```

Este script:
- ✓ Carga el modelo desde caché local
- ✓ Aplica cuantización 4-bit (~1-2 min)
- ✓ Genera una respuesta de prueba
- ✓ Usa solo ~5.6GB de RAM

### 2. Jupyter Notebook (Para experimentar)

```bash
jupyter lab week3/day4.ipynb
```

El notebook incluye:
- Carga del modelo con cuantización
- Ejemplos de generación de texto
- Demo de persistencia offline
- Verificación de caché

### 3. Script Interactivo Personalizado

```python
import os
import torch
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuración
load_dotenv(find_dotenv(".env", usecwd=True), override=True)
HF_TOKEN = os.getenv("HF_TOKEN")
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=HF_TOKEN)

# Configurar cuantización
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Cargar modelo (toma 1-2 min)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config,
    token=HF_TOKEN,
    trust_remote_code=True
)

# Generar texto
messages = [
    {"role": "system", "content": "Eres un asistente útil."},
    {"role": "user", "content": "Tu pregunta aquí"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(response)
```

## Tiempos de Carga

| Operación | Tiempo |
|-----------|--------|
| Primera vez (descarga) | 5-30 min (depende de internet) |
| Cargar desde caché | ~1-2 min (cuantización) |
| Solo tokenizer | <1 segundo |
| Generar respuesta | 1-10 seg (depende de longitud) |

## Uso de Memoria

- **Modelo original (FP16)**: ~15 GB (en disco)
- **Modelo cuantizado (4-bit)**: ~5.6 GB (en RAM durante uso)
- **Sin cuantización (FP16)**: ~16 GB (en RAM)

## Modo Offline

El modelo **ya funciona offline** porque está en caché:

```bash
# Forzar modo offline (sin internet)
TRANSFORMERS_OFFLINE=1 python demo_llama_simple.py
```

## Parámetros de Generación Útiles

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,      # Máximo de tokens a generar
    temperature=0.7,         # Creatividad (0.1=conservador, 1.0=creativo)
    top_p=0.9,              # Nucleus sampling
    top_k=50,               # Top-k sampling
    do_sample=True,         # Activar sampling
    repetition_penalty=1.1, # Penalizar repeticiones
    pad_token_id=tokenizer.eos_token_id
)
```

## Ejemplos de Uso

### Chat Simple

```python
messages = [
    {"role": "system", "content": "Eres un asistente de programación."},
    {"role": "user", "content": "Explica qué es un closure en Python"}
]
```

### Traducción

```python
messages = [
    {"role": "system", "content": "Eres un traductor profesional."},
    {"role": "user", "content": "Traduce al inglés: 'La inteligencia artificial está transformando el mundo'"}
]
```

### Análisis de Código

```python
messages = [
    {"role": "system", "content": "Eres un experto en revisión de código."},
    {"role": "user", "content": "Revisa este código y sugiere mejoras:\n\n```python\ndef calc(x,y):\n  return x+y\n```"}
]
```

### Generación Creativa

```python
messages = [
    {"role": "system", "content": "Eres un escritor creativo."},
    {"role": "user", "content": "Escribe un haiku sobre machine learning"}
]
```

## Troubleshooting

### "Out of memory" / "CUDA out of memory"

**Solución 1**: Asegúrate de usar cuantización 4-bit
```python
quantization_config=quant_config  # No omitir esto
```

**Solución 2**: Reduce `max_new_tokens`
```python
max_new_tokens=100  # En lugar de 500
```

**Solución 3**: Limpia memoria entre generaciones
```python
import torch
torch.cuda.empty_cache()
```

### Modelo genera texto raro o repetitivo

**Solución**: Ajusta parámetros
```python
temperature=0.8,           # Aumentar creatividad
repetition_penalty=1.2,    # Penalizar repeticiones
top_p=0.95                 # Nucleus sampling
```

### Carga muy lenta

**Verificar**: ¿Está descargando de internet?
```bash
# Monitorear red
nethogs

# Forzar uso de caché
local_files_only=True
```

## Verificar Estado

```bash
# Verificar configuración completa
python verify_setup.py

# Ver modelo en caché
ls -lh ~/.cache/huggingface/hub/models--meta-llama*/

# Ver uso de disco
du -sh ~/.cache/huggingface/
```

## Scripts Disponibles

| Script | Propósito | Tiempo |
|--------|-----------|--------|
| `demo_llama_simple.py` | Demo rápida con inferencia | ~2 min |
| `verify_setup.py` | Verificar configuración | <10 seg |
| `hf_check.py` | Test de autenticación HF | <5 seg |
| `test_model_access.py` | Test sin descargar pesos | <5 seg |
| `save_model_offline.py` | Guardar snapshot (opcional) | 5-30 min |
| `load_model_offline.py` | Demo carga offline | ~2 min |

## Próximos Pasos

1. ✅ **Ya hiciste**: Modelo descargado y funcionando
2. 🎯 **Prueba**: `python demo_llama_simple.py`
3. 📓 **Experimenta**: `jupyter lab week3/day4.ipynb`
4. 🔧 **Personaliza**: Crea tus propios scripts basados en los ejemplos

## Recursos

- **Documentación completa**: `OFFLINE_USAGE.md`
- **Setup offline**: `PERSISTENCIA_RESUMEN.md`
- **Notebook**: `week3/day4.ipynb`
- **Modelo en HF**: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

---

**¡El modelo está listo para usar! 🚀**
