import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from numpy.linalg import cond

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50      # Filas (y=altura) x Columnas (x=ancho)
H = 8               # Paso de celda (h)
VY_TEST = 0.1       # Valor de V_y (vorticidad/convección vertical)
MAX_ITER = 5000     # Iteraciones para asegurar convergencia (Richardson necesita más)
TOLERANCE = 1e-8
V0_INITIAL = 1.0    # Velocidad de entrada (valor de color 1.0)
OMEGA = 0.01        # Parámetro de relajación para Richardson (valor pequeño para convergencia)

# --- DEFINICIÓN DE LAS VIGAS (USANDO ÍNDICES BASE 0) ---
# Viga Inferior (j=0, 1; i=20 a 29)
VIGA_INF_Y_MIN, VIGA_INF_Y_MAX = 0, 2
VIGA_INF_X_MIN, VIGA_INF_X_MAX = 20, 30

# Viga Superior (j=4; i=40 a 49)
VIGA_SUP_Y_MIN, VIGA_SUP_Y_MAX = 4, 5
VIGA_SUP_X_MIN, VIGA_SUP_X_MAX = 40, 50

class FlujoRichardson:
    def __init__(self, omega=OMEGA):
        self._incognita_map = {} 
        self._preparar_mapa_incognitas()

        self.V_k = self._inicializar_matriz_velocidades(V0_INITIAL)
        self.N_INCÓGNITAS = len(self._incognita_map)
        self.omega = omega  # Parámetro de relajación

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

    def calcular_jacobiano(self, V_current):
        """Calcula la matriz Jacobiana para el estado actual."""
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        
        m = 0
        for j in range(NY):
            for i in range(NX):
                
                if not self._es_incognita(i, j):
                    continue

                # Valores de Vecinos
                V_c = V_current[j, i]
                V_r = V_current[j, i + 1]
                V_l = V_current[j, i - 1]
                
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
                
        return J.tocsr()

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

    def calcular_residuo(self, V_current):
        """Calcula el residuo F(V) para cada incógnita."""
        residuos = np.zeros(self.N_INCÓGNITAS)
        
        m = 0
        for j in range(NY):
            for i in range(NX):
                if not self._es_incognita(i, j):
                    continue
                
                # Valores de vecinos
                V_c = V_current[j, i]
                V_r = V_current[j, i + 1]
                V_l = V_current[j, i - 1]
                V_u = V_current[j + 1, i]
                V_d = V_current[j - 1, i]
                
                # Ecuación discretizada (no lineal):
                # F = 4*V_c - (V_r + V_l + V_u + V_d) + 4*V_c*(V_r - V_l) + 4*VY_TEST*(V_u - V_d)
                residuos[m] = (4 * V_c 
                              - (V_r + V_l + V_u + V_d) 
                              + 4 * V_c * (V_r - V_l) 
                              + 4 * VY_TEST * (V_u - V_d))
                m += 1
        
        return residuos

    def solve(self):
        """Método de Richardson con relajación para resolver el sistema no lineal.
        
        Iteración de Richardson: V^(k+1) = V^(k) - omega * F(V^(k))
        donde F(V) es el residuo del sistema de ecuaciones no lineales.
        """
        V_matrix = self.V_k.copy()
        
        for k in range(1, MAX_ITER + 1):
            # Calcular el residuo F(V^k)
            residuos = self.calcular_residuo(V_matrix)
            
            # Actualizar valores usando Richardson: V^(k+1) = V^(k) - omega * F(V^(k))
            m = 0
            V_new_matrix = V_matrix.copy()
            max_cambio = 0
            
            for j in range(NY):
                for i in range(NX):
                    if not self._es_incognita(i, j):
                        continue
                    
                    V_old = V_matrix[j, i]
                    # Actualización de Richardson
                    V_new = V_old - self.omega * residuos[m]
                    
                    # Clipar para mantener límites físicos
                    V_new = np.clip(V_new, 0, V0_INITIAL)
                    
                    V_new_matrix[j, i] = V_new
                    
                    # Calcular cambio máximo
                    cambio = abs(V_new - V_old)
                    max_cambio = max(max_cambio, cambio)
                    
                    m += 1
            
            V_matrix = V_new_matrix
            
            # Imprimir progreso cada 50 iteraciones o si converge
            if k % 50 == 0 or max_cambio < TOLERANCE:
                norma_residuo = np.linalg.norm(residuos)
                print(f"Iteración {k}: Cambio máximo = {max_cambio:.10f} | ||Residuo|| = {norma_residuo:.10f}")

            if max_cambio < TOLERANCE:
                print(f"✅ Convergencia alcanzada en {k} iteraciones.")
                
                # Calcular número de condición de la Jacobiana al converger
                print("\nCalculando número de condición de la Jacobiana...")
                J = self.calcular_jacobiano(V_matrix)
                J_dense = J.toarray()
                numero_condicion = cond(J_dense)
                print(f"📊 Número de condición de la Jacobiana: {numero_condicion:.2e}")
                print(f"🎯 Parámetro de relajación omega usado: {self.omega}")
                break
        
        return V_matrix

# --- FUNCIÓN DE VISUALIZACIÓN ESTÁTICA (MAPA DE CALOR) ---
def plot_solution(V_final, vy_value, omega):
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

    ax.set_title(f'Solución Final de V_x (Richardson, ω={omega}) | Vy={vy_value}')
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
    solver = FlujoRichardson(omega=OMEGA)
    print(f"--- INICIANDO SIMULACIÓN CON RICHARDSON (Vy={VY_TEST}, ω={OMEGA}) ---")
    print(f"Número total de incógnitas a resolver: {solver.N_INCÓGNITAS}")
    print(f"ℹ️  El método de Richardson usa la iteración: V^(k+1) = V^(k) - ω * F(V^(k))")
    print(f"ℹ️  donde F(V) es el residuo del sistema no lineal")
    
    V_solution = solver.solve()
    
    print("\nSimulación completada. Generando mapa de calor de la solución final (convergencia)...")
    plot_solution(V_solution, VY_TEST, OMEGA)
