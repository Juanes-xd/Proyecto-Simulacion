import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50  # Filas (y=altura) x Columnas (x=ancho)
H = 8  # Paso de celda (h)
VY_TEST = 0.1  # Valor de V_y (vorticidad/convección vertical)
MAX_ITER = 30
TOLERANCE = 1e-8
V0_INITIAL = 1.0

# --- DEFINICIÓN DE LAS VIGAS (USANDO ÍNDICES BASE 0) ---
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

        # Condiciones de Frontera
        V_matrix[NY - 1, :] = V0_INITIAL  # Fila 4 (Techo)
        V_matrix[:, 0] = V0_INITIAL  # Columna 0 (Entrada)
        V_matrix[0, :] = 0.0  # Fila 0 (Pared inferior)
        V_matrix[:, NX - 1] = 0.0  # Columna 49 (Salida)

        # Vigas (V=0.0)
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0

        # Inicialización de Incógnitas con degradado vertical
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
                V_c = V_current[j, i]
                V_r = V_current[j, i + 1]
                V_l = V_current[j, i - 1]
                V_u = V_current[j + 1, i] if j + 1 < NY else V_c
                V_d = V_current[j - 1, i] if j - 1 >= 0 else V_c

                # Residual F
                F_val = (4 * V_c - (V_r + V_l + V_u + V_d)
                         + 4 * V_c * (V_r - V_l)
                         + 4 * VY_TEST * (V_u - V_d))
                F[m] = F_val

                # --- Ensamblaje del Jacobiano (J) ---
                # Central
                J[m, m] = 4 + 4 * (V_r - V_l)

                # Derecha
                n_r = self._map_to_linear_index(i + 1, j)
                if n_r is not None:
                    J[m, n_r] = -1 + 4 * V_c

                # Izquierda
                n_l = self._map_to_linear_index(i - 1, j)
                if n_l is not None:
                    J[m, n_l] = -1 - 4 * V_c

                # Superior
                n_u = self._map_to_linear_index(i, j + 1)
                if n_u is not None:
                    J[m, n_u] = -1 + 4 * VY_TEST

                # Inferior
                n_d = self._map_to_linear_index(i, j - 1)
                if n_d is not None:
                    J[m, n_d] = -1 - 4 * VY_TEST

                m += 1

        return J.tocsr(), F


def plot_jacobian_sparsity(J):
    """Visualiza el patrón de esparcidad de la matriz Jacobiana."""
    plt.figure(figsize=(8, 8))
    J_coo = J.tocoo()
    plt.plot(J_coo.col, J_coo.row, 'k.', markersize=2)
    plt.title('Patrón de Esparcidad de la Matriz Jacobiana (J)')
    plt.xlabel('Columna (Índice de Incógnita n)')
    plt.ylabel('Fila (Índice de Ecuación m)')
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.show()


def plot_v0(V_initial, vy_value):
    fig, ax = plt.subplots(figsize=(12, 4))
    cax = ax.imshow(V_initial, cmap='viridis', origin='lower',
                    extent=[0, NX, 0, NY], vmin=0, vmax=V0_INITIAL)
    cbar = fig.colorbar(cax, ax=ax, label='Valor de Velocidad (Vx)')
    cbar.set_ticks(np.linspace(0, V0_INITIAL, 6))

    # Dibujar las vigas (Obstáculos)
    ax.add_patch(plt.Rectangle((VIGA_INF_X_MIN, VIGA_INF_Y_MIN),
                               VIGA_INF_X_MAX - VIGA_INF_X_MIN,
                               VIGA_INF_Y_MAX - VIGA_INF_Y_MIN,
                               color='red', alpha=0.8, fill=True))
    ax.add_patch(plt.Rectangle((VIGA_SUP_X_MIN, VIGA_SUP_Y_MIN),
                               VIGA_SUP_X_MAX - VIGA_SUP_X_MIN,
                               VIGA_SUP_Y_MAX - VIGA_SUP_Y_MIN,
                               color='red', alpha=0.8, fill=True))

    ax.set_title(f'Campo de velocidades V0 inicial | Vy={vy_value}')
    ax.set_xlabel('Índice de Columna (x)')
    ax.set_ylabel('Índice de Fila (y)')
    ax.set_xticks(range(0, NX + 1, 5))
    ax.set_yticks(range(0, NY))
    ax.set_ylim(0, NY)

    plt.show()


# --- EJECUCIÓN ---
if __name__ == '__main__':
    solver = FlujoNewtonRaphson()
    print(f"--- FASE DE ENSAMBLAJE (Vy={VY_TEST}) ---")
    print(f"Número total de incógnitas a resolver: {solver.N_INCÓGNITAS}")

    # 1. Ensamblar la matriz J y el vector F usando la matriz inicial V_k
    J_initial, F_initial = solver.ensamblar_FJ(solver.V_k)

    # 2. Mostrar información sobre el Jacobiano
    print(f"\n✅ Ensamblaje completado para V0.")
    print(f"Dimensiones del Jacobiano (J): {J_initial.shape}")
    print(f"Número de elementos no nulos en J: {J_initial.nnz}")

    # 3. Visualización de V0 (para contexto)
    print("\nGenerando mapa de calor de la condición inicial (V0)...")
    plot_v0(solver.V_k, VY_TEST)

    # 4. Visualización del Jacobiano
    print("Generando patrón de esparcidad del Jacobiano...")
    plot_jacobian_sparsity(J_initial)
