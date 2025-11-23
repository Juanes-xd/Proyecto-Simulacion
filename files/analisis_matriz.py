import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
from numpy.linalg import cond, norm, eig
import seaborn as sns

# Importar la clase del sistema
import sys
sys.path.append('.')
from campo_velocidadesV2 import FlujoNewtonRaphson, NY, NX, VY_TEST

print("="*80)
print("ANÁLISIS DE LA MATRIZ JACOBIANA DEL SISTEMA")
print("="*80)

# Crear solver y obtener solución inicial
solver = FlujoNewtonRaphson()
V_initial = solver.V_k.copy()

# Iterar unas pocas veces para obtener un estado representativo
print("\nObteniendo matriz Jacobiana en estado representativo...")
for _ in range(5):
    J, rhs = solver.ensamblar_sistema_newton(V_initial)
    Delta_V = np.zeros(solver.N_INCÓGNITAS)
    from scipy.sparse.linalg import spsolve
    Delta_V = spsolve(J, rhs)
    
    m = 0
    for j in range(NY):
        for i in range(NX):
            if solver._es_incognita(i, j):
                V_initial[j, i] += 0.6 * Delta_V[m]
                m += 1
    V_initial = np.clip(V_initial, 0, 1.0)

# Obtener matriz final
J, _ = solver.ensamblar_sistema_newton(V_initial)
J_dense = J.toarray()
n = J_dense.shape[0]

print(f"\n{'='*80}")
print(f"PROPIEDADES DE LA MATRIZ JACOBIANA (tamaño {n}×{n})")
print(f"{'='*80}")

# 1. NÚMERO DE CONDICIÓN
print(f"\n1️⃣  NÚMERO DE CONDICIÓN")
print(f"   {'─'*70}")
kappa = cond(J_dense)
print(f"   κ(J) = {kappa:.2e}")
if kappa < 100:
    print(f"   ✅ Bien condicionada (κ < 100)")
elif kappa < 1000:
    print(f"   ⚠️  Moderadamente mal condicionada (100 < κ < 1000)")
else:
    print(f"   ❌ Mal condicionada (κ > 1000)")
print(f"   Impacto: Todos los métodos pueden sufrir pérdida de precisión")

# 2. SIMETRÍA
print(f"\n2️⃣  SIMETRÍA")
print(f"   {'─'*70}")
es_simetrica = np.allclose(J_dense, J_dense.T, rtol=1e-10)
norma_asimetria = norm(J_dense - J_dense.T, 'fro') / norm(J_dense, 'fro')
print(f"   ||J - J^T||_F / ||J||_F = {norma_asimetria:.2e}")
if es_simetrica:
    print(f"   ✅ Matriz SIMÉTRICA")
else:
    print(f"   ❌ Matriz NO SIMÉTRICA (asimetría: {norma_asimetria:.2e})")
print(f"   Impacto: Métodos que requieren simetría NO son aplicables")

# 3. DIAGONAL DOMINANCIA
print(f"\n3️⃣  DIAGONAL DOMINANCIA")
print(f"   {'─'*70}")
diagonal_dom_filas = 0
diagonal_dom_estricta = 0
for i in range(n):
    suma_fila = np.sum(np.abs(J_dense[i, :])) - np.abs(J_dense[i, i])
    if np.abs(J_dense[i, i]) >= suma_fila:
        diagonal_dom_filas += 1
        if np.abs(J_dense[i, i]) > suma_fila:
            diagonal_dom_estricta += 1

porcentaje_dd = (diagonal_dom_filas / n) * 100
print(f"   Filas con diagonal dominante: {diagonal_dom_filas}/{n} ({porcentaje_dd:.1f}%)")
print(f"   Diagonal dominante estricta: {diagonal_dom_estricta}/{n}")

if diagonal_dom_filas == n:
    print(f"   ✅ DIAGONAL DOMINANTE (todas las filas)")
elif porcentaje_dd > 80:
    print(f"   ⚠️  Casi diagonal dominante ({porcentaje_dd:.1f}%)")
else:
    print(f"   ❌ NO diagonal dominante ({porcentaje_dd:.1f}%)")

print(f"   Impacto: Jacobi y Gauss-Seidel requieren diagonal dominancia para convergencia")

# 4. DEFINIDA POSITIVA
print(f"\n4️⃣  DEFINICIÓN POSITIVA")
print(f"   {'─'*70}")
eigenvalues = eig(J_dense)[0]
eigenvalues_real = np.real(eigenvalues)
min_eigenval = np.min(eigenvalues_real)
max_eigenval = np.max(eigenvalues_real)
num_positivos = np.sum(eigenvalues_real > 0)
num_negativos = np.sum(eigenvalues_real < 0)

print(f"   Autovalores: min = {min_eigenval:.4f}, max = {max_eigenval:.4f}")
print(f"   Positivos: {num_positivos}, Negativos: {num_negativos}")

if es_simetrica:
    if num_negativos == 0:
        print(f"   ✅ DEFINIDA POSITIVA (todos los autovalores > 0)")
    elif num_positivos == 0:
        print(f"   DEFINIDA NEGATIVA (todos los autovalores < 0)")
    else:
        print(f"   ❌ INDEFINIDA (autovalores mixtos)")
else:
    print(f"   ⚠️  NO aplicable (matriz no simétrica)")

print(f"   Impacto: Gradiente Conjugado requiere definida positiva")

# 5. RADIO ESPECTRAL
print(f"\n5️⃣  RADIO ESPECTRAL Y CONVERGENCIA")
print(f"   {'─'*70}")
radio_espectral = np.max(np.abs(eigenvalues))
print(f"   Radio espectral ρ(J) = {radio_espectral:.4f}")

# Para Jacobi: necesitamos ρ(D^(-1)(L+U)) < 1
D = np.diag(np.diag(J_dense))
D_inv = np.linalg.inv(D)
L_U = J_dense - D
M_jacobi = -D_inv @ L_U
rho_jacobi = np.max(np.abs(eig(M_jacobi)[0]))
print(f"   ρ(M_jacobi) = {rho_jacobi:.4f}")

if rho_jacobi < 1:
    print(f"   ✅ Jacobi debería converger (ρ < 1)")
else:
    print(f"   ❌ Jacobi NO convergerá (ρ ≥ 1)")

# Para Gauss-Seidel
L = np.tril(J_dense, -1)
U = np.triu(J_dense, 1)
D_L_inv = np.linalg.inv(D + L)
M_gs = -D_L_inv @ U
rho_gs = np.max(np.abs(eig(M_gs)[0]))
print(f"   ρ(M_gauss-seidel) = {rho_gs:.4f}")

if rho_gs < 1:
    print(f"   ✅ Gauss-Seidel debería converger (ρ < 1)")
else:
    print(f"   ❌ Gauss-Seidel NO convergerá (ρ ≥ 1)")

# 6. ESTRUCTURA DE LA MATRIZ
print(f"\n6️⃣  ESTRUCTURA Y SPARSIDAD")
print(f"   {'─'*70}")
num_nonzeros = np.count_nonzero(J_dense)
sparsity = (1 - num_nonzeros / (n * n)) * 100
print(f"   Elementos no nulos: {num_nonzeros} de {n*n}")
print(f"   Sparsidad: {sparsity:.2f}%")
print(f"   ✅ Matriz DISPERSA (sparse) - bien para métodos iterativos")

# Ancho de banda
bandwidth = 0
for i in range(n):
    for j in range(n):
        if J_dense[i, j] != 0:
            bandwidth = max(bandwidth, abs(i - j))
print(f"   Ancho de banda: {bandwidth}")
print(f"   ✅ Estructura de banda - apropiada para métodos iterativos")

print(f"\n{'='*80}")
print("CONCLUSIONES: MÉTODOS QUE SE PUEDEN DESCARTAR A PRIORI")
print(f"{'='*80}\n")

metodos_descartados = []
metodos_viables = []

# Análisis por método
print("📋 ANÁLISIS POR MÉTODO:\n")

# Newton-Raphson
print("1. NEWTON-RAPHSON")
print("   Requisitos: Jacobiana calculable, preferible bien condicionada")
if kappa < 1e6:
    print("   ✅ VIABLE - κ aceptable, usa Jacobiana completa")
    metodos_viables.append("Newton-Raphson")
else:
    print("   ⚠️  DIFÍCIL - mal condicionada")
    metodos_descartados.append(("Newton-Raphson", "Matriz muy mal condicionada"))

# Gauss-Seidel
print("\n2. GAUSS-SEIDEL")
print("   Requisitos: Diagonal dominante O matriz simétrica definida positiva")
puede_gs = False
razon_gs = []
if diagonal_dom_filas == n:
    print("   ✅ Cumple: Diagonal dominante")
    puede_gs = True
elif es_simetrica and num_negativos == 0:
    print("   ✅ Cumple: Simétrica definida positiva")
    puede_gs = True
else:
    if not (diagonal_dom_filas == n):
        razon_gs.append("NO diagonal dominante")
    if not (es_simetrica and num_negativos == 0):
        razon_gs.append("NO simétrica def. positiva")

if rho_gs >= 1:
    razon_gs.append(f"Radio espectral ρ={rho_gs:.2f} ≥ 1")
    puede_gs = False

if puede_gs:
    print("   ✅ VIABLE")
    metodos_viables.append("Gauss-Seidel")
else:
    print(f"   ❌ DESCARTAR - {', '.join(razon_gs)}")
    metodos_descartados.append(("Gauss-Seidel", ', '.join(razon_gs)))

# Jacobi
print("\n3. JACOBI")
print("   Requisitos: Diagonal dominante estricta O convergencia demostrada")
puede_jacobi = False
razon_jacobi = []
if diagonal_dom_estricta == n:
    print("   ✅ Cumple: Diagonal dominante estricta")
    puede_jacobi = True
else:
    razon_jacobi.append(f"NO diagonal dominante estricta ({diagonal_dom_estricta}/{n})")

if rho_jacobi >= 1:
    razon_jacobi.append(f"Radio espectral ρ={rho_jacobi:.2f} ≥ 1")
    puede_jacobi = False

if puede_jacobi:
    print("   ✅ VIABLE")
    metodos_viables.append("Jacobi")
else:
    print(f"   ❌ DESCARTAR - {', '.join(razon_jacobi)}")
    metodos_descartados.append(("Jacobi", ', '.join(razon_jacobi)))

# Richardson
print("\n4. RICHARDSON")
print("   Requisitos: Parámetro ω bien elegido, matriz con autovalores acotados")
if not es_simetrica:
    print(f"   ⚠️  DIFÍCIL - matriz no simétrica dificulta elección de ω óptimo")
    metodos_descartados.append(("Richardson", "No simétrica, difícil elegir ω óptimo"))
else:
    print("   ⚠️  Requiere ajuste cuidadoso de ω")
    metodos_viables.append("Richardson (con ω ajustado)")

# Gradiente Conjugado
print("\n5. GRADIENTE CONJUGADO")
print("   Requisitos: Simétrica Y definida positiva")
puede_cg = es_simetrica and num_negativos == 0
if puede_cg:
    print("   ✅ VIABLE")
    metodos_viables.append("Gradiente Conjugado")
else:
    razones_cg = []
    if not es_simetrica:
        razones_cg.append("NO simétrica")
    if num_negativos > 0:
        razones_cg.append("NO definida positiva")
    print(f"   ❌ DESCARTAR - {', '.join(razones_cg)}")
    metodos_descartados.append(("Gradiente Conjugado", ', '.join(razones_cg)))

# Resumen final
print(f"\n{'='*80}")
print("📊 RESUMEN FINAL")
print(f"{'='*80}\n")

print("❌ MÉTODOS A DESCARTAR:")
if metodos_descartados:
    for metodo, razon in metodos_descartados:
        print(f"   • {metodo}")
        print(f"     Razón: {razon}")
else:
    print("   Ninguno (todos son teóricamente viables)")

print("\n✅ MÉTODOS VIABLES:")
for metodo in metodos_viables:
    print(f"   • {metodo}")

print(f"\n{'='*80}")
print("💡 RECOMENDACIÓN:")
print(f"{'='*80}")
print("""
Para este sistema NO LINEAL con matriz Jacobiana que cambia en cada iteración:

✅ MEJOR OPCIÓN: Newton-Raphson
   - Usa la información completa de la Jacobiana
   - Convergencia cuadrática cerca de la solución
   - Maneja bien la no linealidad

⚠️  OPCIONES SECUNDARIAS: 
   - Métodos iterativos solo si se modifica la formulación o se usa
     relajación (SOR) con parámetros cuidadosamente elegidos

❌ EVITAR: Jacobi, Gauss-Seidel, Richardson sin relajación
   - No cumplen criterios de convergencia estándar
   - Como vimos en los experimentos: se estancan y oscilan
""")

# Visualización de la matriz
print("\nGenerando visualización de la estructura de la matriz...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Estructura de sparsidad
ax1 = axes[0]
J_binary = (np.abs(J_dense) > 1e-10).astype(int)
ax1.spy(J_binary, markersize=2, color='black')
ax1.set_title(f'Estructura de Sparsidad\n({num_nonzeros} elementos no nulos, {sparsity:.1f}% sparse)', fontsize=12)
ax1.set_xlabel('Columna')
ax1.set_ylabel('Fila')

# Mapa de calor de valores
ax2 = axes[1]
im = ax2.imshow(J_dense, cmap='RdBu_r', aspect='auto', vmin=-10, vmax=10)
ax2.set_title(f'Valores de la Matriz Jacobiana\nκ(J) = {kappa:.2e}', fontsize=12)
ax2.set_xlabel('Columna')
ax2.set_ylabel('Fila')
plt.colorbar(im, ax=ax2, label='Valor')

plt.tight_layout()
plt.savefig('analisis_matriz_jacobiana.png', dpi=150, bbox_inches='tight')
print("✅ Visualización guardada en: analisis_matriz_jacobiana.png")
plt.show()
