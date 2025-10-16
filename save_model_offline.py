#!/usr/bin/env python3
"""
Script para guardar modelo Llama 3.1 para uso offline.

IMPORTANTE: Los modelos cuantizados con BitsAndBytes NO se pueden guardar
directamente. Este script:
1. Guarda el tokenizer
2. Crea snapshot del modelo original (FP16/BF16)
3. Documenta cómo cargar offline con re-cuantización

Uso:
    python save_model_offline.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import login, snapshot_download
from transformers import AutoTokenizer

# Configuración
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
ARTIFACTS_DIR = Path("./artifacts/llama31-8b")
MIRROR_DIR = Path("./hf_mirror")

# Cargar token
env_path = find_dotenv(".env", usecwd=True)
load_dotenv(env_path, override=True)
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("❌ HF_TOKEN no encontrado en .env")
    exit(1)

login(HF_TOKEN)

print("=" * 70)
print("GUARDANDO MODELO LLAMA 3.1 PARA USO OFFLINE")
print("=" * 70)

# 1. Guardar tokenizer
print(f"\n[1/2] Guardando tokenizer en {ARTIFACTS_DIR}...")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=HF_TOKEN)
tokenizer.save_pretrained(ARTIFACTS_DIR)
print(f"✓ Tokenizer guardado")
print(f"  Archivos: {list(ARTIFACTS_DIR.glob('*'))[:5]}...")

# 2. Crear snapshot local completo
print(f"\n[2/2] Descargando snapshot completo del modelo...")
print(f"  Destino: {MIRROR_DIR}")
print("  ⚠️  ADVERTENCIA: Esto descargará ~16GB (modelo completo en FP16)")
print("  El proceso puede tomar varios minutos...")

try:
    snapshot_path = snapshot_download(
        repo_id=LLAMA,
        token=HF_TOKEN,
        cache_dir=str(MIRROR_DIR),
        local_dir=str(MIRROR_DIR / "llama31-8b"),
        local_dir_use_symlinks=False,  # Copiar archivos reales, no symlinks
        resume_download=True
    )
    print(f"✓ Snapshot descargado en: {snapshot_path}")

    # Obtener revision/commit
    from huggingface_hub import hf_hub_download
    revision_file = MIRROR_DIR / "llama31-8b" / ".git" / "HEAD"
    if not revision_file.exists():
        # Intentar obtener de HF
        import requests
        response = requests.get(
            f"https://huggingface.co/api/models/{LLAMA}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"}
        )
        if response.ok:
            revision = response.json().get("sha", "unknown")
            print(f"  Revision: {revision[:8]}")

except Exception as e:
    print(f"❌ Error descargando snapshot: {e}")
    print("\nSi la descarga falla, el modelo se cargará desde el caché de HuggingFace")
    print("ubicado en ~/.cache/huggingface/hub/")

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"\n✓ Tokenizer guardado: {ARTIFACTS_DIR}")
print(f"✓ Snapshot del modelo: {MIRROR_DIR / 'llama31-8b'}")

print("\n📝 NOTAS IMPORTANTES:")
print("1. Los modelos 4-bit NO se pueden guardar con save_pretrained()")
print("2. El snapshot contiene el modelo original (FP16)")
print("3. La cuantización 4-bit se aplica AL CARGAR, no al guardar")
print("4. Para uso offline: ver load_model_offline.py")

print("\n🔍 Caché de HuggingFace:")
print(f"  HF_HOME: {os.getenv('HF_HOME', '~/.cache/huggingface')}")
print(f"  También se cachea automáticamente en: ~/.cache/huggingface/hub/")

print("\n✅ Configuración para uso offline completada")
print("   Siguiente paso: python load_model_offline.py")
