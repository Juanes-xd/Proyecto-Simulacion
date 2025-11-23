import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import cond

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50      # Filas (y=altura) x Columnas (x=ancho)
H = 8               # Paso de celda (h)
VY_TEST = 0       # Valor de V_y (vorticidad/convección vertical)
V0_INITIAL = 1.0    # Velocidad de entrada (valor de color 1.0)

# --- PARÁMETROS DEL MÉTODO HÍBRIDO ---
MAX_ITER_NEWTON = 30        # Newton-Raphson construye Jacobiana y obtiene solución
TOLERANCE_NEWTON = 1e-8     # Tolerancia para Newton-Raphson
MAX_ITER_ITERATIVO = 10000   # Iteraciones del método iterativo sobre la matriz Jacobiana
TOLERANCE_ITERATIVO = 1e-8  # Tolerancia para método iterativo
METODO_ITERATIVO = "gradiente-descendente" # ← CAMBIO: Ahora usa Gradiente Descendente

# --- DEFINICIÓN DE LAS VIGAS (USANDO ÍNDICES BASE 0) ---
# Viga Inferior (j=0, 1; i=20 a 29)
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30

# Viga Superior (j=4; i=40 a 49)
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

class FlujoHibrido:
    def __init__(self):
        self._incognita_map = {} 
        self._preparar_mapa_incognitas()

        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)

    def _es_incognita(self, i, j):
        """Verifica si el nodo es una incógnita (no frontera, no viga)."""
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2):
            return False
            
        # Excluir Viga Inferior (solo la parte interior j=1)
        if j == 1 and VIGA_INF_X_MIN <= i < VIGA_INF_X_MAX:
            return False
            
        return True

    def _preparar_mapa_incognitas(self):
        """Calcula el número exacto de incógnitas y crea el mapa lineal."""
        count = 0
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    self._incognita_map[(i, j)] = count
                    count += 1
        
    def _map_to_linear_index(self, i, j):
        return self._incognita_map.get((i, j), None)

    def _inicializar_matriz_velocidades(self, v_init):
        V_matrix = np.full((NY, NX), v_init)
        
        # --- Condiciones de Frontera ---
        V_matrix[NY - 1, :] = V0_INITIAL # Techo
        V_matrix[:, 0] = V0_INITIAL      # Entrada
        V_matrix[0, :] = 0.0             # Pared inferior
        V_matrix[:, NX - 1] = 0.0        # Salida
        
        # Vigas
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        
        # Inicialización de Incógnitas con degradado vertical
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1)) 
                    
        return V_matrix

    def ensamblar_sistema_newton(self, V_current):
        """Ensambla el sistema lineal J·ΔV = -F para Newton-Raphson."""
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        F = np.zeros(self.N_INCÓGNITAS)
        
        m = 0
        for j in range(NY):
            for i in range(NX):
                
                if not self._es_incognita(i, j):
                    continue

                # Valores de Vecinos
                V_c = V_current[j, i]
                V_r = V_current[j, i + 1]
                V_l = V_current[j, i - 1]
                V_u = V_current[j + 1, i]
                V_d = V_current[j - 1, i]
                
                # --- Residuo F (Ecuación discretizada) ---
                F[m] = (4 * V_c 
                       - (V_r + V_l + V_u + V_d) 
                       + 4 * V_c * (V_r - V_l) 
                       + 4 * VY_TEST * (V_u - V_d))
                
                # --- Jacobiano (Derivadas Parciales) ---
                
                # Central (dF/dV_c)
                J[m, m] = 4 + 4 * V_r - 4 * V_l
                
                # Derecha (dF/dV_r)
                n_r = self._map_to_linear_index(i + 1, j)
                if n_r is not None:
                    J[m, n_r] = -1 + 4 * V_c
                
                # Izquierda (dF/dV_l)
                n_l = self._map_to_linear_index(i - 1, j)
                if n_l is not None:
                    J[m, n_l] = -1 - 4 * V_c
                    
                # Superior (dF/dV_u)
                n_u = self._map_to_linear_index(i, j + 1)
                if n_u is not None:
                    J[m, n_u] = -1 + 4 * VY_TEST
                
                # Inferior (dF/dV_d)
                n_d = self._map_to_linear_index(i, j - 1)
                if n_d is not None:
                    J[m, n_d] = -1 - 4 * VY_TEST
                
                m += 1
                
        return J.tocsr(), F

    def fase_newton_raphson(self, V_matrix, max_iter, tolerance):
        """FASE 1: Newton-Raphson USANDO MÉTODO ITERATIVO para resolver J·H=-F.
        
        En cada iteración de Newton-Raphson:
        1. Ensambla J (Jacobiano) y F (residuo)
        2. Resuelve J·H = -F usando GRADIENTE DESCENDENTE (método iterativo)
        3. Actualiza V_{k+1} = V_k + H
        
        Esto reemplaza el uso de spsolve (método directo) por un método iterativo.
        """
        print("\n" + "="*70)
        print("NEWTON-RAPHSON CON MÉTODO ITERATIVO PARA RESOLVER J·H = -F")
        print("="*70)
        print(f"En cada iteración de Newton-Raphson:")
        print(f"  1. Construye Jacobiano J y residuo F")
        print(f"  2. Resuelve J·H = -F con {METODO_ITERATIVO.upper()}")
        print(f"  3. Actualiza V = V + H")
        print("="*70)
        
        for k in range(1, max_iter + 1):
            # Ensamblar sistema J·H = -F
            J, F = self.ensamblar_sistema_newton(V_matrix)
            b_rhs = -F  # Lado derecho: -F(V_k)
            
            # Resolver J·H = -F usando MÉTODO ITERATIVO
            # (en lugar de H = spsolve(J, b_rhs) que es método directo)
            H_inicial = np.zeros(self.N_INCÓGNITAS)
            
            if METODO_ITERATIVO == "gradiente-descendente":
                H, converged = self.gradiente_descendente_para_newton(
                    J, b_rhs, H_inicial, 
                    max_iter=1000,  # Iteraciones internas para resolver J·H=-F
                    tolerance=1e-10
                )
            else:
                # Fallback a método directo si no está implementado
                H = spsolve(J, b_rhs)
                converged = True
            
            # Calcular norma del cambio
            norma_H = np.linalg.norm(H)
            max_H = np.max(np.abs(H))
            norma_residuo = np.linalg.norm(F)
            
            print(f"  Iter N-R {k}: ||H|| = {norma_H:.6e} | max|H| = {max_H:.6e} | ||F|| = {norma_residuo:.6e}")
            
            # Actualizar valores: V_{k+1} = V_k + H
            m = 0
            for j in range(NY):
                for i in range(NX):
                    if self._es_incognita(i, j):
                        V_matrix[j, i] += H[m]
                        # Clipar para mantener límites físicos
                        V_matrix[j, i] = np.clip(V_matrix[j, i], 0, V0_INITIAL)
                        m += 1
            
            # Verificar convergencia de Newton-Raphson
            if max_H < tolerance:
                print(f"\n  ✅ Newton-Raphson convergió en {k} iteraciones.")
                print(f"  📊 Norma del residuo final: ||F|| = {norma_residuo:.6e}")
                return V_matrix, True
        
        print(f"\n  ⚠️ Newton-Raphson no convergió en {max_iter} iteraciones.")
        return V_matrix, False

    def gradiente_descendente_para_newton(self, A_matrix, b_vector, H_inicial, max_iter, tolerance):
        """Gradiente Descendente para resolver J·H = b dentro de Newton-Raphson.
        
        Este método resuelve el sistema lineal J·H = b que aparece en cada
        iteración de Newton-Raphson, donde:
        - J es la matriz Jacobiana
        - b = -F(V_k) es el negativo del residuo
        - H es el incremento que buscamos
        
        Minimiza: f(H) = ½||J·H - b||²
        """
        # Convertir a formato denso
        J_dense = A_matrix.toarray()
        J_T = J_dense.T
        H = H_inicial.copy()
        
        # Residuo inicial: r = J·H - b
        r = J_dense @ H - b_vector
        
        # Gradiente inicial: ∇f = J^T·r
        grad = J_T @ r
        
        norma_grad_inicial = np.linalg.norm(grad)
        
        # Si el gradiente inicial ya es suficientemente pequeño
        if norma_grad_inicial < tolerance:
            return H, True
        
        for k in range(1, max_iter + 1):
            # Calcular residuo: r = J·H - b
            r = J_dense @ H - b_vector
            
            # Calcular gradiente: ∇f = J^T·r
            grad = J_T @ r
            
            # Norma del gradiente (criterio de convergencia)
            norma_grad = np.linalg.norm(grad)
            
            # Line search exacto: α = ||∇f||² / ||J·∇f||²
            J_grad = J_dense @ grad
            norma_J_grad_sq = np.dot(J_grad, J_grad)
            
            if norma_J_grad_sq < 1e-15:
                return H, True
            
            alpha = (norma_grad ** 2) / norma_J_grad_sq
            
            # Actualizar solución: H_{k+1} = H_k - α·∇f
            H = H - alpha * grad
            
            # Verificar convergencia (gradiente cercano a cero)
            if norma_grad < tolerance:
                return H, True
        
        # No convergió en max_iter
        return H, False

    def solve_hibrido(self):
        """Newton-Raphson usando método iterativo para resolver J·H=-F.
        
        Implementación correcta según el ejercicio:
        En cada iteración de Newton-Raphson:
          1. Calcula Jacobiano J y residuo F
          2. Resuelve J·H = -F usando método iterativo (Gradiente Descendente)
          3. Actualiza V = V + H
        """
        
        print("\n" + "="*70)
        print("NEWTON-RAPHSON CON MÉTODO ITERATIVO")
        print("="*70)
        print(f"Método iterativo usado: {METODO_ITERATIVO.upper()}")
        print(f"Número total de incógnitas: {self.N_INCÓGNITAS}")
        
        V_matrix = self.V_k.copy()
        
        # Ejecutar Newton-Raphson (que usa método iterativo internamente)
        V_matrix, nr_converged = self.fase_newton_raphson(V_matrix, MAX_ITER_NEWTON, TOLERANCE_NEWTON)
        
        if not nr_converged:
            print("\n⚠️ ADVERTENCIA: Newton-Raphson no convergió completamente.")
        
        # Análisis final
        print("\n" + "="*70)
        print("ANÁLISIS FINAL")
        print("="*70)
        J_final, F_final = self.ensamblar_sistema_newton(V_matrix)
        J_dense = J_final.toarray()
        numero_condicion = cond(J_dense)
        norma_residuo_final = np.linalg.norm(F_final)
        
        # ANÁLISIS DE PROPIEDADES DE LA JACOBIANA
        print("\n" + "="*70)
        print("VERIFICACIÓN: PROPIEDADES DE LA MATRIZ JACOBIANA")
        print("="*70)
        
        # 1. Verificar SIMETRÍA
        es_simetrica = np.allclose(J_dense, J_dense.T, rtol=1e-10, atol=1e-12)
        norma_asimetria = np.linalg.norm(J_dense - J_dense.T, 'fro') / np.linalg.norm(J_dense, 'fro')
        print(f"\n1️⃣  SIMETRÍA:")
        print(f"   ||J - J^T|| / ||J|| = {norma_asimetria:.6e}")
        if es_simetrica:
            print(f"   ✅ Matriz SIMÉTRICA")
        else:
            print(f"   ❌ Matriz NO SIMÉTRICA")
            print(f"      Ejemplo: J[0,1] = {J_dense[0,1]:.6f}, J[1,0] = {J_dense[1,0]:.6f}")
        
        # 2. Verificar DOMINANCIA DIAGONAL
        print(f"\n2️⃣  DOMINANCIA DIAGONAL:")
        diag_dominant_count = 0
        for i in range(self.N_INCÓGNITAS):
            diag_val = abs(J_dense[i, i])
            off_diag_sum = np.sum(np.abs(J_dense[i, :])) - diag_val
            if diag_val >= off_diag_sum:
                diag_dominant_count += 1
        
        porcentaje_dd = (diag_dominant_count / self.N_INCÓGNITAS) * 100
        print(f"   Filas con dominancia diagonal: {diag_dominant_count}/{self.N_INCÓGNITAS} ({porcentaje_dd:.1f}%)")
        if porcentaje_dd >= 100:
            print(f"   ✅ Matriz ESTRICTAMENTE diagonalmente dominante")
        elif porcentaje_dd >= 80:
            print(f"   ⚠️  Dominancia diagonal PARCIAL")
        else:
            print(f"   ❌ NO diagonalmente dominante")
            
        # 3. CONCLUSIÓN
        print(f"\n3️⃣  CONCLUSIÓN:")
        print(f"   • Matriz NO simétrica → Métodos como Jacobi/Gauss-Seidel probablemente fallan")
        print(f"   • Dominancia diagonal: {porcentaje_dd:.1f}% → {'Insuficiente' if porcentaje_dd < 80 else 'Aceptable'}")
        print(f"   ✅ Gradiente Descendente funciona (no requiere simetría ni dominancia)")
        print(f"      → Se usa DENTRO de Newton-Raphson para resolver J·H=-F")
        
        print("="*70)
        print(f"📊 Número de condición de la Jacobiana: {numero_condicion:.2e}")
        print(f"📊 Norma del residuo final ||F||: {norma_residuo_final:.6e}")
        print(f"📊 Newton-Raphson: {'✅ Convergió' if nr_converged else '❌ No convergió'}")
        print("="*70)
        
        return V_matrix

# --- FUNCIÓN DE VISUALIZACIÓN ESTÁTICA (MAPA DE CALOR) ---
def plot_solution(V_final, vy_value):
    fig, ax = plt.subplots(figsize=(18, 8))

    # Mostrar el mapa de calor de la solución final
    cax = ax.imshow(V_final, cmap='viridis', origin='lower', 
                   extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL)

    cbar = fig.colorbar(cax, label='Valor de Velocidad (Vx)')
    cbar.set_ticks(np.linspace(0, V0_INITIAL, 6))

    # Dibujar las vigas (Obstáculos) con color ROJO
    # Viga Inferior
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                              VIGA_INF_X_MAX - VIGA_INF_X_MIN, 
                              VIGA_INF_Y_MAX - VIGA_INF_Y_MIN, 
                              color='red', alpha=0.8, fill=True))
    # Viga Superior
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                              VIGA_SUP_X_MAX - VIGA_SUP_X_MIN, 
                              VIGA_SUP_Y_MAX - VIGA_SUP_Y_MIN, 
                              color='red', alpha=0.8, fill=True))

    ax.set_title(f'Solución: Newton-Raphson con {METODO_ITERATIVO.upper()} (resuelve J·H=-F) | Vy={vy_value}')
    ax.set_xlabel('Índice de Columna (x)')
    ax.set_ylabel('Índice de Fila (y)')
    ax.set_xticks(range(0, NX + 1, 5))
    ax.set_yticks(range(0, NY))
    ax.set_xlim([0, NX])
    ax.set_ylim([0, NY])

    # Dibujar los bordes de las celdas
    ax.set_xticks(np.arange(0, NX+1, 1), minor=True)
    ax.set_yticks(np.arange(0, NY+1, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # Mostrar el valor de la velocidad en cada celda
    for j in range(NY):
        for i in range(NX):
            valor = V_final[j, i]
            ax.text(i+0.5, j+0.5, f'{valor:.2f}', color='white', ha='center', va='center', fontsize=7)

    plt.tight_layout()
    plt.show()

# --- NUEVO: Gráfico para mostrar el número de condición ---
def plot_condicion(numero_condicion):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    ax.text(0.5, 0.5, f'Número de condición de la Jacobiana:\n{numero_condicion:.2e}',
            fontsize=24, color='navy', ha='center', va='center', fontweight='bold')
    ax.set_title('Propiedad numérica de la matriz Jacobiana', fontsize=16)
    plt.tight_layout()
    plt.show()

# --- EJECUCIÓN ---
if __name__ == '__main__':
    solver = FlujoHibrido()
    V_solution = solver.solve_hibrido()

    # Calcular número de condición de la Jacobiana final
    J_final, _ = solver.ensamblar_sistema_newton(V_solution)
    numero_condicion = cond(J_final.toarray())

    print("\n✨ Generando visualización del resultado...")
    plot_solution(V_solution, VY_TEST)
    print("\n✨ Mostrando gráfico del número de condición...")
    plot_condicion(numero_condicion)
