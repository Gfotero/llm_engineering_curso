#!/usr/bin/env python3
"""
Test rápido de acceso al modelo Llama 3.1 (solo config, sin descargar pesos).
"""
import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from transformers import AutoConfig, AutoTokenizer

# Cargar .env
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN no encontrado")
    exit(1)

print(f"✓ HF_TOKEN encontrado")

# Login
login(HF_TOKEN)
print("✓ Login exitoso")

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Test 1: Cargar config (muy ligero)
try:
    config = AutoConfig.from_pretrained(LLAMA, token=HF_TOKEN)
    print(f"✓ Config cargado - hidden_size: {config.hidden_size}")
except Exception as e:
    print(f"❌ Error cargando config: {e}")
    exit(1)

# Test 2: Cargar tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=HF_TOKEN)
    print(f"✓ Tokenizer cargado - vocab_size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ Error cargando tokenizer: {e}")
    exit(1)

print("\n✅ Tienes acceso completo al modelo Llama 3.1")
print(f"   Para cargar el modelo completo, usa el notebook week3/day4.ipynb")
print(f"   NOTA: Requiere GPU con CUDA o ~16GB RAM con cuantización")
