def factorial(n: int) -> int:
    if type(n) is not int:
        raise TypeError("El input debe ser un entero.")
    if n < 0:
        raise ValueError("El input no puede ser un número negativo.")
    if n == 0:
        return 1

    result = 1
    for i in range(1, n + 1):
        result *= i
    return result