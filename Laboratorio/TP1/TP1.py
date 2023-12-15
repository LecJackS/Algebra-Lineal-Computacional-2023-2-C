import numpy as np
from scipy.linalg import solve_triangular
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time


def desarmarMatriz(A):
    """
    Descompone una matriz A en sus 4 bloques principales.

    Args:
        A (numpy.ndarray): La matriz de entrada.

    Returns:
        Tuple: Una tupla con los siguientes elementos en orden: u11, U12, L21, A22.
    """
    n = A.shape[0]
    u11 = A[0, 0]
    U12 = A[0:1, 1:n]
    L21 = A[1:n, 0:1] / u11
    A22 = A[1:n, 1:n]
    return u11, U12, L21, A22

def armarMatriz(u11, U12, L21, L22, U22):
    """
    Construye la descomposición LU de una matriz a partir de sus componentes.

    Args:
        u11: Primer elemento diagonal.
        U12: Elementos superiores de la primera fila.
        L21: Elementos inferiores de la primera columna.
        L22: Matriz triangular inferior.
        U22: Matriz triangular superior.

    Returns:
        numpy.ndarray: La descomposición LU de la matriz correspondiente.
    """
    n = L22.shape[0] + 1
    L = np.eye(n)
    U = np.zeros((n, n))

    L[1:n, 0:1] = L21
    L[1:n, 1:n] = L22

    U[0, 0] = u11
    U[0:1, 1:n] = U12
    U[1:n, 1:n] = U22

    #  Use np.block to merge the matrices
    # n = L22.shape[0] + 1
    # L = np.block([[1, np.zeros((1, n - 1))], [L21, L22]])
    # U = np.block([[u11, U12], [np.zeros((n - 1, 1)), U22]])

    return L, U

def es_identidad(A):
    """
    Devuelve True si la matriz A es la matriz identidad.
    Reemplaza np.allclose() de manera rápida, pues no se necesita tolerancia.
    """
    es_I = np.abs(A - np.eye(A.shape[0])).sum() == 0
    return es_I



@profile
def computarLU(A):
    n = A.shape[0]  # Tamaño de la matriz cuadrada A
    if n == 1:
        # Caso base: A ya es una matriz de 1x1, por lo que L y U son simplemente escalares.
        return np.array([[1.0]]), A
    
    # Dividir A en bloques
    a11 = A[0, 0]
    A12 = A[0:1, 1:]
    A21 = A[1:, 0:1]
    A22 = A[1:, 1:]
    
    # Calcula u11, U12 y L21
    u11 = a11
    U12 = A12
    L21 = A21 / u11
    
    L22U22 = A22 - L21 @ U12 #np.outer(L21, U12)
    # Llamada recursiva para calcular L22 y U22
    L22, U22 = computarLU(L22U22)
    
    # Transponer L21 y U12 antes de la concatenación
    #L21 = L21[:, np.newaxis]  # Transponer L21 a un vector columna
    #U12 = U12[np.newaxis, :]  # Transponer U12 a un vector fila
    
    # Combinar los bloques L y U
    # Combinar los bloques L y U de manera eficiente
    # Combinar los bloques L y U de manera eficiente
    # Combinar los bloques L y U de manera eficiente
    # Combinar los bloques L y U de manera eficiente
    # Combinar los bloques L y U de manera eficiente
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    L[0, 0] = 1.0
    L[1:n, 0:1] = L21
    L[1:n, 1:n] = L22

    U[0, 0] = u11
    U[0:1, 1:n] = U12
    U[1:n, 1:n] = U22


    
    return L, U

@profile
def resolverLU(A, b):
    """TODO: Docstring for resolverLU."""
    
    L, U = computarLU(A)
    
    try:
        y = solve_triangular(L, b, lower=True)
        x = solve_triangular(U, y)
    except Exception as e:
        print(e)
        print("L:", L)
        print("U:", U)
        assert False
        x = np.zeros_like(b)

    return x

def calcularError(A, b, x):
    """ """
    e = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    return e

@profile
def inversa(A):
    """Calcula la inversa de una matriz A utilizando la factorización LU"""
    n = A.shape[0]
    I = np.eye(n)
    A_inv = np.zeros((n, n))
    for i in range(n):
        x = resolverLU(A, I[:, i])
        A_inv[:, i] = x
    return A_inv

@profile
def experimento(min_dim=100, max_dim=200, iteraciones=1, step=10):
    dimensiones = np.arange(min_dim, max_dim, step)

    # Las primeras 20 posiciones se llenan con ceros para respetar que el indice
    # del vector se corresponda con la dimension de las matrices del experimento
    v1, v2 = [0]*min_dim, [0]*min_dim

    times_LU = []
    times_Inv = []

    for n in tqdm(dimensiones):
        sum_e1 = 0
        sum_e2 = 0
        for i in range(iteraciones):
            A = np.random.uniform(low=-1., high=1., size=(n, n))
            x = np.random.uniform(low=-1., high=1., size=(n, 1))
            b = A @ x

            time_0 = time()
            x1 = resolverLU(A, b)
            e1 = calcularError(A, b, x1)
            e1 = np.log(e1)
            times_LU.append(time()-time_0)

            time_0 = time()
            x2 = inversa(A) @ b
            e2 = calcularError(A, b, x2)
            e2 = np.log(e2)
            times_Inv.append(time()-time_0)

            sum_e1 += e1
            sum_e2 += e2
        v1.append(sum_e1)
        v2.append(sum_e2)
    
    return v1, v2, times_LU, times_Inv



if __name__ == "__main__":
    v1, v2, timesLU, timesInv = experimento()

    # plt.plot(v1, label="LU")
    # plt.plot(v2, label="Inversa")
    # plt.legend()
    # plt.show()

    # plt.plot(timesLU, label="LU")
    # plt.plot(timesInv, label="Inversa")
    # plt.legend()
    # plt.show()

