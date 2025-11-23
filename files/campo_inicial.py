import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50      # Filas (y=altura) x Columnas (x=ancho)
H = 8               # Paso de celda (h)
VY_TEST = 0.1       # Valor de V_y (vorticidad/convección vertical)
MAX_ITER = 30       # Iteraciones para asegurar convergencia
TOLERANCE = 1e-8
V0_INITIAL = 1.0    # Velocidad de entrada (valor de color 1.0)

# --- DEFINICIÓN DE LAS VIGAS (USANDO ÍNDICES BASE 0) ---
# Viga Inferior (j=0, 1; i=20 a 29)
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30

# Viga Superior (j=4; i=40 a 49)
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

class FlujoNewtonRaphson:
    def __init__(self):
        self._incognita_map = {} 
        self._preparar_mapa_incognitas()

        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)

    def _es_incognita(self, i, j):
        """Verifica si el nodo es una incógnita (no frontera, no viga)."""
        # Candidatos a incógnita: Nodos INTERIORES (i in [1, NX-2], j in [1, NY-2])
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
        
        # 1. Aplicar V=1.0 para Techo y Entrada (Donde sea aplicable)
        V_matrix[NY - 1, :] = V0_INITIAL # Fila 4 completa (Techo)
        V_matrix[:, 0] = V0_INITIAL      # Columna 0 (Entrada, excluyendo la esquina (0,0))

        # 2. Aplicar V=0.0 para Pared Inferior, Salida y Esquina (Prioridad a V=0.0)
        V_matrix[0, :] = 0.0             # Fila 0 completa (Pared inferior)
        V_matrix[:, NX - 1] = 0.0        # Columna 49 completa (Salida)
        # La esquina (0,0) queda en 0.0 por la prioridad de la fila 0
        
        # 3. Vigas (También son V=0.0)
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        
        # 4. Inicialización de Incógnitas con degradado vertical
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1)) 
                    
        return V_matrix

    def ensamblar_FJ(self, V_current):
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        F = np.zeros(self.N_INCÓGNITAS)
        
        m = 0
        for j in range(NY):
            for i in range(NX):
                
                if not self._es_incognita(i, j):
                    continue

                # Valores de Vecinos
                V_c = V_current[j, i]; V_r = V_current[j, i + 1]
                V_l = V_current[j, i - 1]; V_u = V_current[j + 1, i]
                V_d = V_current[j - 1, i]
                
                # Residual F (Adaptado de la fórmula F_i,j = 0)
                F_val = (4 * V_c 
                         - (V_r + V_l + V_u + V_d) 
                         + 4 * V_c * (V_r - V_l) 
                         + 4 * VY_TEST * (V_u - V_d)
                        )
                F[m] = F_val

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

    def solve(self):
        V_matrix = self.V_k.copy()
        
        for k in range(1, MAX_ITER + 1):
            J, F = self.ensamblar_FJ(V_matrix)
            residual_norm = np.linalg.norm(F)
            
            # --- Paso de Newton-Raphson: J * Delta_V = -F ---
            Delta_V_vector = spsolve(J, -F)
            
            # --- Actualización con amortiguamiento (0.6) y clipado ---
            m = 0
            V_new_matrix = V_matrix.copy()
            for j in range(NY):
                for i in range(NX):
                    if self._es_incognita(i, j):
                        V_new_matrix[j, i] = V_matrix[j, i] + 0.6 * Delta_V_vector[m]
                        m += 1
            
            V_matrix = np.clip(V_new_matrix, 0, V0_INITIAL)
            
            print(f"Iteración {k}: Norma del Residual = {residual_norm:.10f}")

            if residual_norm < TOLERANCE:
                print(f"✅ Convergencia alcanzada en {k} iteraciones.")
                break
        
        return V_matrix

# --- FUNCIÓN DE VISUALIZACIÓN ESTÁTICA (MAPA DE CALOR) ---
def plot_solution(V_final, vy_value):
    fig, ax = plt.subplots(figsize=(12, 4))
    
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
    
    ax.set_title(f'Solución Final de V_x (Newton-Raphson) | Vy={vy_value}')
    ax.set_xlabel('Índice de Columna (x)')
    ax.set_ylabel('Índice de Fila (y)')
    ax.set_xticks(range(0, NX + 1, 5))
    ax.set_yticks(range(0, NY))
    ax.set_ylim(0, NY)
    
    plt.show()

# --- EJECUCIÓN ---
if __name__ == '__main__':
    solver = FlujoNewtonRaphson()
    print(f"--- INICIANDO SIMULACIÓN ESTÁTICA (Vy={VY_TEST}) ---")
    print(f"Número total de incógnitas a resolver: {solver.N_INCÓGNITAS}")
    
    V_solution = solver.solve()
    
    print("\nSimulación completada. Generando mapa de calor de la solución final (convergencia)...")
    plot_solution(V_solution, VY_TEST)