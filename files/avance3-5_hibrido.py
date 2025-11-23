import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numpy.linalg import cond

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50      # Filas (y=altura) x Columnas (x=ancho)
H = 8               # Paso de celda (h)
VY_TEST = 0.1       # Valor de V_y (vorticidad/convección vertical)
V0_INITIAL = 1.0    # Velocidad de entrada (valor de color 1.0)

# --- PARÁMETROS DEL MÉTODO HÍBRIDO ---
MAX_ITER_NEWTON = 30        # Newton-Raphson construye Jacobiana y obtiene solución
TOLERANCE_NEWTON = 1e-8     # Tolerancia para Newton-Raphson
MAX_ITER_ITERATIVO = 5000   # Iteraciones del método iterativo sobre la matriz Jacobiana
TOLERANCE_ITERATIVO = 1e-8  # Tolerancia para método iterativo
METODO_ITERATIVO = "jacobi" # Opciones: "jacobi" o "gauss-seidel"

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
        """FASE 1: Newton-Raphson converge COMPLETAMENTE hasta obtener solución."""
        print("\n" + "="*70)
        print("FASE 1: NEWTON-RAPHSON (Convergencia COMPLETA)")
        print("="*70)
        print("Objetivo: Obtener una solución convergida que servirá como")
        print("          condición inicial para el método iterativo.")
        
        for k in range(1, max_iter + 1):
            # Ensamblar sistema J·ΔV = -F
            J, F = self.ensamblar_sistema_newton(V_matrix)
            
            # Resolver sistema lineal
            delta_V = spsolve(J, -F)
            
            # Calcular norma del cambio
            norma_delta = np.linalg.norm(delta_V)
            norma_residuo = np.linalg.norm(F)
            max_delta = np.max(np.abs(delta_V))
            
            print(f"  Iteración {k}: ||ΔV|| = {norma_delta:.6e} | max|ΔV| = {max_delta:.6e} | ||F|| = {norma_residuo:.6e}")
            
            # Actualizar valores
            m = 0
            for j in range(NY):
                for i in range(NX):
                    if self._es_incognita(i, j):
                        V_matrix[j, i] += delta_V[m]
                        # Clipar para mantener límites físicos
                        V_matrix[j, i] = np.clip(V_matrix[j, i], 0, V0_INITIAL)
                        m += 1
            
            # Verificar convergencia
            if max_delta < tolerance:
                print(f"\n  ✅ Newton-Raphson convergió COMPLETAMENTE en {k} iteraciones.")
                print(f"  📊 Esta solución se usará como punto de partida para Gauss-Seidel.")
                return V_matrix, True
        
        print(f"\n  ⚠️ Newton-Raphson no convergió en {max_iter} iteraciones.")
        return V_matrix, False

    def iteracion_jacobi_matriz(self, J_matrix, b_vector, V_inicial, max_iter, tolerance):
        """Método de Jacobi PURO aplicado al sistema lineal J·V = b.
        
        Iteración de Jacobi: V^(k+1) = D^(-1) * (b - (L+U)*V^k)
        donde J = D + L + U (diagonal + triangular inferior + triangular superior)
        """
        print("\n" + "="*70)
        print(f"FASE 2: MÉTODO DE JACOBI PURO SOBRE LA MATRIZ JACOBIANA")
        print("="*70)
        print("Resolviendo J·V = b usando iteración de Jacobi (sin relajación)")
        print("donde J es la Jacobiana del sistema no lineal.\n")
        
        # Convertir a formato denso para manipulación
        J_dense = J_matrix.toarray()
        V_k = V_inicial.copy()
        
        # Extraer diagonal D, parte inferior L, y parte superior U
        D = np.diag(np.diag(J_dense))
        D_inv = np.diag(1.0 / np.diag(J_dense))  # Inversa de D
        L_plus_U = J_dense - D  # L + U
        
        for k in range(1, max_iter + 1):
            # Jacobi: V^(k+1) = D^(-1) * (b - (L+U)*V^k)
            V_k_new = D_inv @ (b_vector - L_plus_U @ V_k)
            
            # Calcular cambio
            cambio = np.linalg.norm(V_k_new - V_k)
            max_cambio = np.max(np.abs(V_k_new - V_k))
            
            # Actualizar
            V_k = V_k_new
            
            # Imprimir progreso
            if k % 100 == 0 or cambio < tolerance:
                residuo = np.linalg.norm(J_dense @ V_k - b_vector)
                print(f"  Iteración {k}: ||V^(k+1)-V^k|| = {cambio:.6e} | max|ΔV| = {max_cambio:.6e} | ||J·V-b|| = {residuo:.6e}")
            
            if cambio < tolerance:
                print(f"\n  ✅ Jacobi convergió en {k} iteraciones.")
                return V_k, True
        
        print(f"\n  ⚠️ Jacobi no convergió en {max_iter} iteraciones.")
        return V_k, False

    def iteracion_gauss_seidel_matriz(self, J_matrix, b_vector, V_inicial, max_iter, tolerance):
        """Método de Gauss-Seidel PURO aplicado al sistema lineal J·V = b.
        
        Iteración de Gauss-Seidel: V^(k+1) = (D+L)^(-1) * (b - U*V^k)
        donde J = D + L + U
        """
        print("\n" + "="*70)
        print(f"FASE 2: MÉTODO DE GAUSS-SEIDEL PURO SOBRE LA MATRIZ JACOBIANA")
        print("="*70)
        print("Resolviendo J·V = b usando iteración de Gauss-Seidel (sin relajación)")
        print("donde J es la Jacobiana del sistema no lineal.\n")
        
        # Convertir a formato denso
        J_dense = J_matrix.toarray()
        V_k = V_inicial.copy()
        n = len(V_k)
        
        for k in range(1, max_iter + 1):
            V_k_old = V_k.copy()
            
            # Gauss-Seidel: Actualización secuencial
            for i in range(n):
                suma = 0
                for j in range(n):
                    if j != i:
                        suma += J_dense[i, j] * V_k[j]
                
                if abs(J_dense[i, i]) > 1e-12:
                    V_k[i] = (b_vector[i] - suma) / J_dense[i, i]
            
            # Calcular cambio
            cambio = np.linalg.norm(V_k - V_k_old)
            max_cambio = np.max(np.abs(V_k - V_k_old))
            
            # Imprimir progreso
            if k % 100 == 0 or cambio < tolerance:
                residuo = np.linalg.norm(J_dense @ V_k - b_vector)
                print(f"  Iteración {k}: ||V^(k+1)-V^k|| = {cambio:.6e} | max|ΔV| = {max_cambio:.6e} | ||J·V-b|| = {residuo:.6e}")
            
            if cambio < tolerance:
                print(f"\n  ✅ Gauss-Seidel convergió en {k} iteraciones.")
                return V_k, True
        
        print(f"\n  ⚠️ Gauss-Seidel no convergió en {max_iter} iteraciones.")
        return V_k, False

    def solve_hibrido(self):
        """Método híbrido: Newton-Raphson construye J → Método iterativo sobre J·V=b."""
        print("\n" + "="*70)
        print("MÉTODO HÍBRIDO: NEWTON-RAPHSON → MÉTODO ITERATIVO SOBRE JACOBIANA")
        print("="*70)
        print(f"Estrategia:")
        print(f"  1. Newton-Raphson converge y obtiene solución V* (hasta {MAX_ITER_NEWTON} iter)")
        print(f"  2. Se construye sistema lineal J·V = b en el punto V*")
        print(f"  3. Método iterativo PURO ({METODO_ITERATIVO.upper()}) resuelve J·V = b sin relajación")
        print(f"\nNúmero total de incógnitas: {self.N_INCÓGNITAS}")
        
        V_matrix = self.V_k.copy()
        
        # FASE 1: Newton-Raphson converge completamente
        V_matrix, nr_converged = self.fase_newton_raphson(V_matrix, MAX_ITER_NEWTON, TOLERANCE_NEWTON)
        
        if not nr_converged:
            print("\n⚠️ ADVERTENCIA: Newton-Raphson no convergió. El resultado puede no ser confiable.")
            return V_matrix
        
        # Extraer vector solución de Newton-Raphson
        V_nr_vector = np.zeros(self.N_INCÓGNITAS)
        m = 0
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    V_nr_vector[m] = V_matrix[j, i]
                    m += 1
        
        # Construir sistema J·V = b en el punto de convergencia
        print("\n" + "="*70)
        print("CONSTRUCCIÓN DEL SISTEMA LINEAL J·V = b")
        print("="*70)
        print("Usando la solución de Newton-Raphson para construir J y b...")
        J_final, F_final = self.ensamblar_sistema_newton(V_matrix)
        b_vector = -F_final  # En Newton-Raphson: J·ΔV = -F, pero aquí queremos J·V = b
        # El vector b correcto es: b = J·V_nr (para que J·V = b tenga solución V = V_nr)
        b_vector = J_final @ V_nr_vector
        
        print(f"✅ Sistema construido: J({self.N_INCÓGNITAS}×{self.N_INCÓGNITAS})·V = b")
        print(f"   Usando V* de Newton-Raphson como condición inicial para iteración")
        
        # FASE 2: Método iterativo sobre la matriz Jacobiana
        if METODO_ITERATIVO == "jacobi":
            V_iter_vector, iter_converged = self.iteracion_jacobi_matriz(
                J_final, b_vector, V_nr_vector, MAX_ITER_ITERATIVO, TOLERANCE_ITERATIVO
            )
        elif METODO_ITERATIVO == "gauss-seidel":
            V_iter_vector, iter_converged = self.iteracion_gauss_seidel_matriz(
                J_final, b_vector, V_nr_vector, MAX_ITER_ITERATIVO, TOLERANCE_ITERATIVO
            )
        else:
            print(f"❌ Método '{METODO_ITERATIVO}' no reconocido")
            return V_matrix
        
        # Reconstruir matriz de velocidades
        V_final_matrix = V_matrix.copy()
        m = 0
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    V_final_matrix[j, i] = V_iter_vector[m]
                    m += 1
        
        # Análisis final
        print("\n" + "="*70)
        print("ANÁLISIS FINAL")
        print("="*70)
        J_check, F_check = self.ensamblar_sistema_newton(V_final_matrix)
        J_dense = J_check.toarray()
        numero_condicion = cond(J_dense)
        norma_residuo_final = np.linalg.norm(F_check)
        
        # Comparar soluciones
        diferencia = np.linalg.norm(V_iter_vector - V_nr_vector)
        
        print(f"📊 Número de condición de la Jacobiana: {numero_condicion:.2e}")
        print(f"📊 Norma del residuo final ||F||: {norma_residuo_final:.6e}")
        print(f"📊 Diferencia ||V_iterativo - V_newton||: {diferencia:.6e}")
        print(f"📊 Newton-Raphson: {'✅ Convergió' if nr_converged else '❌ No convergió'}")
        print(f"📊 {METODO_ITERATIVO.capitalize()}: {'✅ Convergió' if iter_converged else '⚠️ No convergió'}")
        
        if nr_converged and iter_converged:
            print(f"\n✅ ÉXITO: Ambos métodos convergieron.")
            if diferencia < 1e-6:
                print(f"   Las soluciones son prácticamente idénticas (diferencia < 1e-6)")
        
        return V_final_matrix

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

    ax.set_title(f'Solución Final (Método Híbrido: N-R → {METODO_ITERATIVO.capitalize()} sobre J) | Vy={vy_value}')
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

# --- EJECUCIÓN ---
if __name__ == '__main__':
    solver = FlujoHibrido()
    V_solution = solver.solve_hibrido()
    
    print("\n✨ Generando visualización del resultado...")
    plot_solution(V_solution, VY_TEST)
