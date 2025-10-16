#!/usr/bin/env python3
"""
Script de carga offline del modelo Llama 3.1 con cuantización 4-bit.

Este script demuestra cómo cargar el modelo SIN conexión a internet,
aplicando cuantización 4-bit en el momento de carga.

PREREQUISITOS:
1. Ejecutar save_model_offline.py primero
2. Tener el modelo cacheado en ~/.cache/huggingface/ o en ./hf_mirror/

Variables de entorno importantes:
- HF_HOME: Directorio base de caché de HuggingFace
- TRANSFORMERS_CACHE: Caché de transformers
- TRANSFORMERS_OFFLINE=1 o HF_HUB_OFFLINE=1: Modo offline estricto

Uso:
    python load_model_offline.py

    # O en modo offline estricto:
    TRANSFORMERS_OFFLINE=1 python load_model_offline.py
"""
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Configuración
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ARTIFACTS_DIR = Path("./artifacts/llama31-8b")
MIRROR_DIR = Path("./hf_mirror/llama31-8b")

print("=" * 70)
print("CARGA OFFLINE DE MODELO LLAMA 3.1 CON CUANTIZACIÓN 4-BIT")
print("=" * 70)

# Verificar modo offline
offline_mode = os.getenv("TRANSFORMERS_OFFLINE") == "1" or os.getenv("HF_HUB_OFFLINE") == "1"
print(f"\nModo offline: {offline_mode}")
print(f"HF_HOME: {os.getenv('HF_HOME', '~/.cache/huggingface')}")
print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', 'default')}")

# Opción 1: Cargar desde artifacts (solo tokenizer)
print(f"\n[Opción 1] Cargando tokenizer desde {ARTIFACTS_DIR}...")
if ARTIFACTS_DIR.exists():
    try:
        tokenizer_local = AutoTokenizer.from_pretrained(
            str(ARTIFACTS_DIR),
            local_files_only=True
        )
        print(f"✓ Tokenizer cargado desde artifacts")
        print(f"  Vocab size: {tokenizer_local.vocab_size}")
    except Exception as e:
        print(f"❌ Error: {e}")
        tokenizer_local = None
else:
    print(f"⚠️  {ARTIFACTS_DIR} no existe. Ejecuta save_model_offline.py primero")
    tokenizer_local = None

# Opción 2: Cargar desde mirror local (modelo completo)
print(f"\n[Opción 2] Cargando desde mirror local {MIRROR_DIR}...")
if MIRROR_DIR.exists():
    model_path = str(MIRROR_DIR)
    print(f"✓ Mirror local encontrado")
else:
    print(f"⚠️  Mirror local no existe, usando caché de HuggingFace")
    model_path = LLAMA

# Opción 3: Cargar desde caché de HuggingFace (más común)
print(f"\n[Opción 3] Cargando desde caché de HuggingFace...")
print("Ubicación: ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/")

# Configurar cuantización 4-bit
print("\nConfigurando cuantización 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
print("✓ BitsAndBytesConfig creado")

# Cargar tokenizer (prioridad: local > caché)
print(f"\nCargando tokenizer...")
try:
    if tokenizer_local:
        tokenizer = tokenizer_local
        print("✓ Usando tokenizer de artifacts")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=offline_mode
        )
        print(f"✓ Tokenizer cargado desde {'mirror' if MIRROR_DIR.exists() else 'caché'}")
except Exception as e:
    print(f"❌ Error cargando tokenizer: {e}")
    print("\nSoluciones:")
    print("1. Ejecuta: python save_model_offline.py")
    print("2. Verifica que el modelo esté en caché: ls ~/.cache/huggingface/hub/")
    print("3. Si estás offline, primero descarga online")
    exit(1)

# Cargar modelo con cuantización
print(f"\nCargando modelo con cuantización 4-bit...")
print("⚠️  NOTA: La cuantización se aplica AL CARGAR, no está pre-guardada")
print("Esto puede tomar 1-2 minutos...")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=quant_config,
        local_files_only=offline_mode,
        trust_remote_code=True
    )
    print(f"✓ Modelo cargado exitosamente")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")
    print(f"  Memory footprint: ~{model.get_memory_footprint() / 1e9:.2f} GB")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    print("\nSi el error es sobre archivos no encontrados:")
    print("1. Verifica que ejecutaste save_model_offline.py")
    print("2. O carga el modelo online primero (sin local_files_only=True)")
    exit(1)

# Test de inferencia
print("\n" + "=" * 70)
print("TEST DE INFERENCIA")
print("=" * 70)

messages = [
    {"role": "system", "content": "Eres un asistente conciso."},
    {"role": "user", "content": "Di 'Funcionando offline' en una frase."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerando respuesta...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\nRespuesta: {response}")

print("\n✅ Carga offline exitosa y modelo funcional")
print(f"   Modo offline: {offline_mode}")
print(f"   Fuente: {'Mirror local' if MIRROR_DIR.exists() else 'Caché HF'}")
