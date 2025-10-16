#!/usr/bin/env python3
"""
Script de verificación completa de la configuración.
Verifica que todo esté listo para uso offline de Llama 3.1.
"""
import os
import sys
from pathlib import Path

print("=" * 70)
print("VERIFICACIÓN DE CONFIGURACIÓN - LLAMA 3.1 OFFLINE")
print("=" * 70)

checks_passed = 0
checks_total = 0

# Check 1: .env y HF_TOKEN
print("\n[1/7] Verificando .env y HF_TOKEN...")
checks_total += 1
try:
    from dotenv import load_dotenv, find_dotenv
    env_path = find_dotenv(".env", usecwd=True)
    load_dotenv(env_path, override=True)
    HF_TOKEN = os.getenv("HF_TOKEN")

    if HF_TOKEN and len(HF_TOKEN) > 10:
        print(f"  ✓ HF_TOKEN encontrado (longitud: {len(HF_TOKEN)})")
        checks_passed += 1
    else:
        print(f"  ❌ HF_TOKEN no válido o no encontrado")
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check 2: Versiones de paquetes
print("\n[2/7] Verificando versiones de paquetes...")
checks_total += 1
try:
    import transformers
    import bitsandbytes
    import torch

    print(f"  transformers: {transformers.__version__}")
    print(f"  bitsandbytes: {bitsandbytes.__version__}")
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check 3: Autenticación HF
print("\n[3/7] Verificando autenticación con HuggingFace...")
checks_total += 1
try:
    from huggingface_hub import login, whoami
    login(HF_TOKEN)
    user_info = whoami()
    print(f"  ✓ Autenticado como: {user_info.get('name', 'unknown')}")
    checks_passed += 1
except Exception as e:
    print(f"  ❌ Error: {e}")

# Check 4: Caché de HuggingFace
print("\n[4/7] Verificando caché de HuggingFace...")
checks_total += 1
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
llama_cache = list(cache_dir.glob("models--meta-llama--Meta-Llama-3.1-8B-Instruct"))

if llama_cache:
    print(f"  ✓ Modelo encontrado en caché: {llama_cache[0].name}")
    checks_passed += 1
else:
    print(f"  ⚠️  Modelo NO encontrado en caché")
    print(f"     Ejecuta: python save_model_offline.py")

# Check 5: Artifacts directory
print("\n[5/7] Verificando artifacts/...")
checks_total += 1
artifacts_dir = Path("./artifacts/llama31-8b")
if artifacts_dir.exists() and list(artifacts_dir.glob("tokenizer*")):
    print(f"  ✓ Tokenizer guardado en {artifacts_dir}")
    print(f"    Archivos: {len(list(artifacts_dir.glob('*')))} archivos")
    checks_passed += 1
else:
    print(f"  ⚠️  Tokenizer no guardado en artifacts/")
    print(f"     Ejecuta: python save_model_offline.py")

# Check 6: Mirror directory (opcional)
print("\n[6/7] Verificando hf_mirror/ (opcional)...")
checks_total += 1
mirror_dir = Path("./hf_mirror/llama31-8b")
if mirror_dir.exists() and list(mirror_dir.glob("*.safetensors")):
    print(f"  ✓ Snapshot local encontrado en {mirror_dir}")
    checks_passed += 1
else:
    print(f"  ⚠️  Snapshot local no encontrado")
    print(f"     (Opcional - se usará caché de HF)")

# Check 7: .gitignore actualizado
print("\n[7/7] Verificando .gitignore...")
checks_total += 1
gitignore_path = Path(".gitignore")
if gitignore_path.exists():
    content = gitignore_path.read_text()
    if "artifacts/" in content and "hf_mirror/" in content:
        print(f"  ✓ .gitignore configurado correctamente")
        checks_passed += 1
    else:
        print(f"  ⚠️  Falta añadir artifacts/ o hf_mirror/ a .gitignore")
else:
    print(f"  ⚠️  .gitignore no encontrado")

# Resumen
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"\nChecks pasados: {checks_passed}/{checks_total}")

if checks_passed >= 5:
    print("\n✅ Configuración CORRECTA - Listo para uso offline")
    print("\nPróximos pasos:")
    print("  1. python load_model_offline.py  # Probar carga offline")
    print("  2. jupyter lab week3/day4.ipynb  # Usar notebook")
    sys.exit(0)
elif checks_passed >= 3:
    print("\n⚠️  Configuración PARCIAL - Algunos componentes faltan")
    print("\nAcciones recomendadas:")
    print("  1. python save_model_offline.py  # Guardar modelo y tokenizer")
    print("  2. Revisar errores arriba")
    sys.exit(1)
else:
    print("\n❌ Configuración INCOMPLETA - Revisar errores")
    print("\nAcciones necesarias:")
    print("  1. Verificar .env con HF_TOKEN")
    print("  2. Ejecutar: python hf_check.py")
    print("  3. Ejecutar: python save_model_offline.py")
    sys.exit(1)
