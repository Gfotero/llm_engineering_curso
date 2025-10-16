#!/usr/bin/env python3
"""
Test de carga del modelo con cuantización 4-bit.
Este script verifica que bitsandbytes funcione correctamente.
"""
import os
import torch
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("=" * 60)
print("Test de carga de modelo con cuantización")
print("=" * 60)

# Verificar versiones
import transformers
import bitsandbytes as bnb
print(f"\ntransformers: {transformers.__version__}")
print(f"bitsandbytes: {bnb.__version__}")
print(f"torch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

# Cargar .env
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("\n❌ HF_TOKEN no encontrado")
    exit(1)

# Login
login(HF_TOKEN)
print("\n✓ Login exitoso")

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Cargar tokenizer
print(f"\nCargando tokenizer de {LLAMA}...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=HF_TOKEN)
print(f"✓ Tokenizer cargado")

# Configurar cuantización 4-bit
print("\nConfigurando cuantización 4-bit...")
try:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    print("✓ BitsAndBytesConfig creado")
except Exception as e:
    print(f"❌ Error creando BitsAndBytesConfig: {e}")
    exit(1)

# Cargar modelo (SOLO si quieres probarlo - comentado por defecto)
print("\n" + "=" * 60)
print("NOTA: La carga del modelo completo está comentada.")
print("Descomenta las líneas abajo si quieres probarlo (descarga ~16GB)")
print("=" * 60)

# Descomentar para probar carga completa:
# print(f"\nCargando modelo {LLAMA} con cuantización 4-bit...")
# print("Esto puede tomar varios minutos...")
# try:
#     model = AutoModelForCausalLM.from_pretrained(
#         LLAMA,
#         device_map="auto",
#         quantization_config=quant_config,
#         token=HF_TOKEN,
#         trust_remote_code=True
#     )
#     print(f"✓ Modelo cargado exitosamente")
#     print(f"  Device: {model.device}")
#     print(f"  Dtype: {model.dtype}")
# except Exception as e:
#     print(f"❌ Error cargando modelo: {e}")
#     import traceback
#     traceback.print_exc()
#     exit(1)

print("\n✅ Configuración validada correctamente")
print("Si quieres cargar el modelo, descomenta las líneas en este script")
