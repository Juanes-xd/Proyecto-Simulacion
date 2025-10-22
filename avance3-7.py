import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from numpy.linalg import norm

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50      # Filas (y=altura) x Columnas (x=ancho)
H = 8               # Paso de celda (h)
VY_TEST = 0.1       # Valor de V_y (vorticidad/convección vertical)
MAX_ITER = 30       # Iteraciones Newton-Raphson
TOLERANCE = 1e-8
V0_INITIAL = 1.0    # Velocidad de entrada (valor de color 1.0)

# Parámetros para Gradiente Conjugado
MAX_ITER_CG = 1000  # Máximo de iteraciones para CG
TOL_CG = 1e-10      # Tolerancia para CG

# --- DEFINICIÓN DE LAS VIGAS ---
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

class FlujoNewtonRaphson:
    def __init__(self):
        self._incognita_map = {} 
        self._preparar_mapa_incognitas()
        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)
        self.historial_condicion = []

    def _es_incognita(self, i, j):
        """Verifica si el nodo es una incógnita (no frontera, no viga)."""
        if not (1 <= i <= NX - 2 and 1 <= j <= NY - 2):
            return False
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
        V_matrix[NY - 1, :] = V0_INITIAL
        V_matrix[:, 0] = V0_INITIAL
        V_matrix[0, :] = 0.0
        V_matrix[:, NX - 1] = 0.0
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1)) 
        return V_matrix

    def ensamblar_sistema_newton(self, V_current):
        """Ensambla el sistema J*H = -F para Newton-Raphson."""
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        F = np.zeros(self.N_INCÓGNITAS)  # Vector de funciones evaluadas
        
        m = 0
        for j in range(NY):
            for i in range(NX):
                if not self._es_incognita(i, j):
                    continue

                V_c = V_current[j, i]
                V_r = V_current[j, i + 1]
                V_l = V_current[j, i - 1]
                V_u = V_current[j + 1, i]
                V_d = V_current[j - 1, i]
                
                # Vector F(x): Ecuación de balance
                F[m] = (4 * V_c 
                       - (V_r + V_l + V_u + V_d) 
                       + 4 * V_c * (V_r - V_l) 
                       + 4 * VY_TEST * (V_u - V_d))

                # Jacobiano (dF/dV)
                J[m, m] = 4 + 4 * V_r - 4 * V_l
                
                n_r = self._map_to_linear_index(i + 1, j)
                if n_r is not None:
                    J[m, n_r] = -1 + 4 * V_c
                
                n_l = self._map_to_linear_index(i - 1, j)
                if n_l is not None:
                    J[m, n_l] = -1 - 4 * V_c
                    
                n_u = self._map_to_linear_index(i, j + 1)
                if n_u is not None:
                    J[m, n_u] = -1 + 4 * VY_TEST
                
                n_d = self._map_to_linear_index(i, j - 1)
                if n_d is not None:
                    J[m, n_d] = -1 - 4 * VY_TEST
                
                m += 1
                
        return J.tocsr(), F

    def gradiente_conjugado(self, A, b, x0=None, tol=TOL_CG, max_iter=MAX_ITER_CG):
        """
        Resuelve A·x = b usando el método del Gradiente Conjugado.
        
        Algoritmo:
        1. r0 = b - A·x0 (residuo inicial)
        2. p0 = r0 (dirección de búsqueda inicial)
        3. Iteración: 
           - α_k = (r_k^T · r_k) / (p_k^T · A · p_k)
           - x_{k+1} = x_k + α_k · p_k
           - r_{k+1} = r_k - α_k · A · p_k
           - β_k = (r_{k+1}^T · r_{k+1}) / (r_k^T · r_k)
           - p_{k+1} = r_{k+1} + β_k · p_k
        """
        n = len(b)
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()
        
        # Residuo inicial: r = b - A·x
        r = b - A.dot(x)
        p = r.copy()  # Dirección de búsqueda inicial
        
        rs_old = np.dot(r, r)
        
        for k in range(max_iter):
            Ap = A.dot(p)
            alpha = rs_old / np.dot(p, Ap)
            
            # Actualizar solución: x = x + α·p
            x = x + alpha * p
            
            # Actualizar residuo: r = r - α·A·p
            r = r - alpha * Ap
            
            rs_new = np.dot(r, r)
            
            # Criterio de convergencia
            if np.sqrt(rs_new) < tol:
                print(f"  CG convergió en {k+1} iteraciones (residuo: {np.sqrt(rs_new):.2e})")
                return x
            
            # Actualizar dirección de búsqueda
            beta = rs_new / rs_old
            p = r + beta * p
            
            rs_old = rs_new
        
        print(f"  CG alcanzó el máximo de iteraciones ({max_iter}). Residuo final: {np.sqrt(rs_new):.2e}")
        return x

    def calcular_numero_condicion(self, J):
        """Calcula el número de condición de la matriz Jacobiana."""
        J_dense = J.toarray()
        # Usar SVD para calcular el número de condición de manera robusta
        U, s, Vt = np.linalg.svd(J_dense, full_matrices=False)
        cond_num = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
        return cond_num

    def solve(self):
        V_matrix = self.V_k.copy()
        
        print("\n" + "="*80)
        print("MÉTODO SELECCIONADO: GRADIENTE CONJUGADO")
        print("="*80)
        print("\nJUSTIFICACIÓN:")
        print("• Se aplica CG al sistema normal J^T·J·H = J^T·(-F)")
        print("• Garantiza matriz simétrica y definida positiva")
        print("• Convergencia rápida: O(√κ) iteraciones")
        print("• Eficiente para sistemas sparse grandes")
        print("\nMÉTODOS DESCARTADOS:")
        print("• Jacobi: Convergencia lenta O(n), requiere diagonal dominante")
        print("• Gauss-Seidel: Mejor que Jacobi pero aún lento, sensible al orden")
        print("• Richardson: Requiere ajuste manual de ω, convergencia lenta")
        print("• Grad. Descendente: Más lento que CG para matrices mal condicionadas")
        print("="*80 + "\n")
        
        for k in range(1, MAX_ITER + 1):
            # Ensamblar sistema J·H = -F
            J, F = self.ensamblar_sistema_newton(V_matrix)
            
            # Calcular número de condición
            numero_condicion = self.calcular_numero_condicion(J)
            self.historial_condicion.append(numero_condicion)
            
            # Lado derecho del sistema: -F(x)
            rhs = -F
            
            print(f"\n--- Iteración Newton-Raphson {k} ---")
            print(f"Número de condición de J: {numero_condicion:.4e}")
            print(f"Norma de F(x): {norm(F):.4e}")
            
            # Resolver J·H = -F usando Gradiente Conjugado
            # Aplicamos CG al sistema normal: (J^T·J)·H = J^T·(-F)
            JT = J.T
            A = JT.dot(J)  # Sistema normal (simétrico y def. positivo)
            b = JT.dot(rhs)
            
            # Resolver con CG
            H = self.gradiente_conjugado(A, b)
            
            # Actualización con amortiguamiento
            damping = 0.6
            m = 0
            V_new_matrix = V_matrix.copy()
            for j in range(NY):
                for i in range(NX):
                    if self._es_incognita(i, j):
                        # X_{i+1} = X_i + damping * H
                        V_new_matrix[j, i] = V_matrix[j, i] + damping * H[m]
                        m += 1
            
            # Aplicar límites físicos
            V_matrix = np.clip(V_new_matrix, 0, V0_INITIAL)
            
            # Criterio de convergencia
            max_cambio = np.max(np.abs(H))
            print(f"Cambio máximo (||H||_∞): {max_cambio:.10f}")

            if max_cambio < TOLERANCE:
                print(f"\n✅ CONVERGENCIA ALCANZADA en {k} iteraciones.")
                break
        
        return V_matrix

# --- FUNCIÓN DE VISUALIZACIÓN DEL CAMPO DE VELOCIDADES ---
def plot_velocity_field(V_final, vy_value):
    """Visualiza el campo de velocidades en una ventana independiente."""
    fig1, ax1 = plt.subplots(figsize=(20, 8))
    
    cax = ax1.imshow(V_final, cmap='viridis', origin='lower', 
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL)
    cbar = fig1.colorbar(cax, ax=ax1, label='Valor de Velocidad (Vx)')
    cbar.set_ticks(np.linspace(0, V0_INITIAL, 6))

    # Dibujar vigas (Obstáculos) con color ROJO
    ax1.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN), 
                               VIGA_INF_X_MAX - VIGA_INF_X_MIN, 
                               VIGA_INF_Y_MAX - VIGA_INF_Y_MIN, 
                               color='red', alpha=0.8, fill=True))
    ax1.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN), 
                               VIGA_SUP_X_MAX - VIGA_SUP_X_MIN, 
                               VIGA_SUP_Y_MAX - VIGA_SUP_Y_MIN, 
                               color='red', alpha=0.8, fill=True))

    ax1.set_title(f'Solución Final de V_x (Newton-Raphson con Gradiente Conjugado) | Vy={vy_value}', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel('Índice de Columna (x)', fontsize=12)
    ax1.set_ylabel('Índice de Fila (y)', fontsize=12)
    ax1.set_xlim([0, NX])
    ax1.set_ylim([0, NY])
    
    # Dibujar los bordes de las celdas
    ax1.set_xticks(np.arange(0, NX+1, 1), minor=True)
    ax1.set_yticks(np.arange(0, NY+1, 1), minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax1.set_xticks(range(0, NX + 1, 5))
    ax1.set_yticks(range(0, NY))
    
    # Mostrar el valor de la velocidad en cada celda
    for j in range(NY):
        for i in range(NX):
            valor = V_final[j, i]
            ax1.text(i+0.5, j+0.5, f'{valor:.2f}', 
                    color='white', ha='center', va='center', 
                    fontsize=8, weight='bold')

    plt.tight_layout()
    plt.show()

# --- FUNCIÓN DE VISUALIZACIÓN DEL NÚMERO DE CONDICIÓN ---
def plot_condition_number(historial_cond):
    """Visualiza la evolución del número de condición en una ventana independiente."""
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    iteraciones = range(1, len(historial_cond) + 1)
    ax2.semilogy(iteraciones, historial_cond, 'o-', 
                linewidth=2.5, markersize=10, color='crimson', 
                markerfacecolor='orange', markeredgewidth=2)
    
    ax2.set_xlabel('Iteración Newton-Raphson', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Número de Condición κ(J)', fontsize=13, fontweight='bold')
    ax2.set_title('Evolución del Número de Condición de la Matriz Jacobiana', 
                 fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=1)
    ax2.set_xticks(iteraciones)
    
    # Añadir anotaciones con mejor formato
    for i, (it, cond) in enumerate(zip(iteraciones, historial_cond)):
        ax2.annotate(f'{cond:.2e}', 
                    xy=(it, cond), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=10,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', 
                             facecolor='yellow', 
                             edgecolor='black',
                             alpha=0.7))
    
    # Añadir información adicional
    ax2.text(0.02, 0.98, 
            f'κ inicial: {historial_cond[0]:.2e}\nκ final: {historial_cond[-1]:.2e}',
            transform=ax2.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.show()

# --- EJECUCIÓN ---
if __name__ == '__main__':
    solver = FlujoNewtonRaphson()
    print(f"\n{'='*80}")
    print(f"SIMULACIÓN DE FLUJO CON MÉTODO DE NEWTON-RAPHSON")
    print(f"{'='*80}")
    print(f"Parámetros de simulación:")
    print(f"  • Malla: {NY} × {NX} nodos")
    print(f"  • Vy (convección vertical): {VY_TEST}")
    print(f"  • Número de incógnitas: {solver.N_INCÓGNITAS}")
    print(f"  • Tolerancia Newton-Raphson: {TOLERANCE}")
    print(f"  • Tolerancia Gradiente Conjugado: {TOL_CG}")
    
    V_solution = solver.solve()
    
    print("\n" + "="*80)
    print("RESULTADOS FINALES")
    print("="*80)
    print(f"Número de condición final: {solver.historial_condicion[-1]:.4e}")
    print(f"Velocidad máxima: {np.max(V_solution):.6f}")
    print(f"Velocidad mínima: {np.min(V_solution):.6f}")
    print("="*80 + "\n")
    
    print("Generando visualizaciones...")
    print("\n[Ventana 1] Campo de Velocidades")
    plot_velocity_field(V_solution, VY_TEST)
    
    print("[Ventana 2] Evolución del Número de Condición")
    plot_condition_number(solver.historial_condicion)