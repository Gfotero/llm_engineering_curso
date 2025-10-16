#!/usr/bin/env python3
"""
Script de validación para autenticación con Hugging Face.
Verifica que el token HF_TOKEN esté configurado y funcione correctamente.
"""
import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

# Cargar variables de entorno
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True)

# Verificar que HF_TOKEN existe
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    print("❌ ERROR: HF_TOKEN no está definido en .env")
    exit(1)
else:
    print(f"✓ HF_TOKEN encontrado (longitud: {len(HF_TOKEN)})")

# Hacer login con huggingface_hub
try:
    login(HF_TOKEN)
    print("✓ Login exitoso con huggingface_hub")
except Exception as e:
    print(f"❌ ERROR en login: {e}")
    exit(1)

# Intentar cargar tokenizer de Llama 3.1
try:
    print("\nIntentando cargar tokenizer de meta-llama/Meta-Llama-3.1-8B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token=HF_TOKEN,
        trust_remote_code=True
    )
    print("✓ OK tokenizer cargado exitosamente")
    print(f"  Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ ERROR al cargar tokenizer: {e}")
    print("\nSugerencia: Verifica que:")
    print("1. Has aceptado la licencia del modelo en: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    print("2. Tu token tiene permisos de lectura")
    print("3. La versión de transformers es compatible")
    exit(1)

print("\n✅ Todas las verificaciones pasaron correctamente")
