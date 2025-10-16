#!/usr/bin/env python3
"""
Demo simple de Llama 3.1 cargando desde caché con cuantización 4-bit.
El modelo ya está descargado, solo se re-cuantiza al cargar.
"""
import os
import torch
from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 70)
print("DEMO LLAMA 3.1 - Cargando desde caché local")
print("=" * 70)

# Cargar token
load_dotenv(find_dotenv(".env", usecwd=True), override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# 1. Cargar tokenizer (rápido)
print("\n[1/3] Cargando tokenizer desde caché...")
tokenizer = AutoTokenizer.from_pretrained(
    LLAMA,
    token=HF_TOKEN,
    local_files_only=True  # Usar solo caché local
)
print(f"✓ Tokenizer cargado (vocab: {tokenizer.vocab_size})")

# 2. Configurar cuantización 4-bit
print("\n[2/3] Configurando cuantización 4-bit...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
print("✓ Configuración creada")

# 3. Cargar modelo desde caché con cuantización
print("\n[3/3] Cargando modelo desde caché con cuantización...")
print("⏳ Esto toma ~1-2 minutos (aplicando cuantización)...")

model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map="auto",
    quantization_config=quant_config,
    token=HF_TOKEN,
    local_files_only=True,  # Usar solo caché local
    trust_remote_code=True
)

print(f"✓ Modelo cargado")
print(f"  Device: {model.device}")
print(f"  Memory: {model.get_memory_footprint() / 1e9:.2f} GB")

# 4. Test de inferencia
print("\n" + "=" * 70)
print("TEST DE INFERENCIA")
print("=" * 70)

messages = [
    {"role": "system", "content": "Eres un asistente útil y conciso."},
    {"role": "user", "content": "Explica en una frase qué es la cuantización de modelos."}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\nGenerando respuesta...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"\n💬 Pregunta: {messages[1]['content']}")
print(f"🤖 Respuesta: {response.strip()}")

print("\n" + "=" * 70)
print("✅ DEMO COMPLETADA - Modelo funcionando correctamente")
print("=" * 70)
print(f"\nEl modelo se cargó desde: ~/.cache/huggingface/hub/")
print(f"Memoria usada: ~{model.get_memory_footprint() / 1e9:.1f}GB (gracias a cuantización 4-bit)")
