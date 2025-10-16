import pytest
from cod_evaluar import factorial

# Caso feliz: Verificación de factorial de un número positivo
def test_factorial_positive():
    assert factorial(5) == 120

# Caso de borde: Verificación del factorial de 0
def test_factorial_zero():
    assert factorial(0) == 1

# Caso de error: Tipo de dato incorrecto (cadena)
def test_factorial_invalid_type_string():
    with pytest.raises(TypeError):
        factorial("5")

# Caso de error: Tipo de dato incorrecto (float)
def test_factorial_invalid_type_float():
    with pytest.raises(TypeError):
        factorial(5.5)

# Caso de error: Número negativo
def test_factorial_negative_number():
    with pytest.raises(ValueError):
        factorial(-1)

# Caso de borde: Verificación para n igual a 1
def test_factorial_one():
    assert factorial(1) == 1

# Caso de borde: Manejo del entero más grande (puede variar según el sistema, aquí verificamos un caso común)
def test_factorial_large_number():
    assert factorial(20) == 2432902008176640000  # 20! = 2432902008176640000