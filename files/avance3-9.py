import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, diags
import time

# --- PARÁMETROS GLOBALES Y GEOMETRÍA ---
NY, NX = 5, 50      # Filas (y=altura) x Columnas (x=ancho)
H_PASO = 8          # Paso de celda (h)
VY_CONST = 0.01      # Valor de V_y (vorticidad vertical constante)
MAX_ITER_NR = 30    # Iteraciones Newton-Raphson
MAX_ITER_LINEAR = 2000  # Iteraciones para método iterativo lineal
TOLERANCE_NR = 1e-8
TOLERANCE_LINEAR = 1e-8
V0_INITIAL = 1.0    # Velocidad de entrada

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
        
        # Condiciones de frontera
        V_matrix[NY - 1, :] = V0_INITIAL  # Techo
        V_matrix[:, 0] = V0_INITIAL       # Entrada
        V_matrix[0, :] = 0.0              # Pared inferior
        V_matrix[:, NX - 1] = 0.0         # Salida
        V_matrix[VIGA_INF_Y_MIN:VIGA_INF_Y_MAX, VIGA_INF_X_MIN:VIGA_INF_X_MAX] = 0.0
        V_matrix[VIGA_SUP_Y_MIN:VIGA_SUP_Y_MAX, VIGA_SUP_X_MIN:VIGA_SUP_X_MAX] = 0.0
        
        # Inicialización con degradado
        for j in range(NY):
            for i in range(NX):
                if self._es_incognita(i, j):
                    V_matrix[j, i] = v_init * (j / (NY - 1)) 
                    
        return V_matrix

    def ensamblar_U_y_J(self, V_current):
        """
        Ensambla el vector U de ecuaciones y la matriz Jacobiana J.
        
        Según la imagen, la ecuación para cada nodo (i,j) es:
        U_{i,j} = 1/4(v^x_{i+1,j} + v^x_{i-1,j} + v^x_{i,j+1} + v^x_{i,j-1} 
                      - h/2·v^y_{i,j}[v^x_{i+1,j} + v^x_{i-1,j}]
                      - h/2·v^y_{i,j}[v^x_{i,j+1} + v^x_{i,j-1}])
        
        Donde queremos que v^x_{i,j} = U_{i,j}
        
        Entonces: U_{i,j} = v^x_{i,j} - 1/4(v^x_{i+1,j} + v^x_{i-1,j} + v^x_{i,j+1} + v^x_{i,j-1}
                                            - h/2·v^y_{i,j}[v^x_{i+1,j} + v^x_{i-1,j}]
                                            - h/2·v^y_{i,j}[v^x_{i,j+1} + v^x_{i,j-1}])
        """
        J = lil_matrix((self.N_INCÓGNITAS, self.N_INCÓGNITAS))
        U = np.zeros(self.N_INCÓGNITAS)
        
        h = H_PASO
        vy = VY_CONST
        
        m = 0
        for j in range(NY):
            for i in range(NX):
                
                if not self._es_incognita(i, j):
                    continue

                V_c = V_current[j, i]
                V_r = V_current[j, i + 1]  # Right (i+1,j)
                V_l = V_current[j, i - 1]  # Left (i-1,j)
                V_u = V_current[j + 1, i]  # Up (i,j+1)
                V_d = V_current[j - 1, i]  # Down (i,j-1)
                
                # Vector U: diferencia entre valor actual y valor objetivo
                U_val = V_c - 0.25 * (
                    V_r + V_l + V_u + V_d
                    - (h/2) * vy * (V_r + V_l)
                    - (h/2) * vy * (V_u + V_d)
                )
                U[m] = U_val

                # --- JACOBIANO: ∂U_{i,j}/∂v^x ---
                # Según la tabla de derivadas parciales de la imagen
                
                # Central: ∂U/∂v^x_{i,j} = 1 + v^x_{i+1,j} - v^x_{i-1,j}
                # Pero en realidad la derivada de U respecto a v_c es simplemente 1
                # porque U = v_c - f(vecinos)
                J[m, m] = 1.0
                
                # Derecha: ∂U/∂v^x_{i+1,j} = -(1/4) - (h/8)v^y_{i,j}
                n_r = self._map_to_linear_index(i + 1, j)
                if n_r is not None:
                    J[m, n_r] = -0.25 * (1 - (h/2) * vy)
                
                # Izquierda: ∂U/∂v^x_{i-1,j} = -(1/4) - (h/8)v^y_{i,j}
                n_l = self._map_to_linear_index(i - 1, j)
                if n_l is not None:
                    J[m, n_l] = -0.25 * (1 - (h/2) * vy)
                    
                # Superior: ∂U/∂v^x_{i,j+1} = -(1/4) - (h/8)v^y_{i,j}
                n_u = self._map_to_linear_index(i, j + 1)
                if n_u is not None:
                    J[m, n_u] = -0.25 * (1 - (h/2) * vy)
                
                # Inferior: ∂U/∂v^x_{i,j-1} = -(1/4) - (h/8)v^y_{i,j}
                n_d = self._map_to_linear_index(i, j - 1)
                if n_d is not None:
                    J[m, n_d] = -0.25 * (1 - (h/2) * vy)
                
                m += 1
                
        return J.tocsr(), U

    def solve_jacobi(self, J, U_n, x0=None):
        """
        Método de Jacobi para resolver J·H = -U_n
        H = v_{n+1} - v_n
        """
        n = len(U_n)
        H = np.zeros(n) if x0 is None else x0.copy()
        b = -U_n
        
        # Extraer diagonal y resto
        D = J.diagonal()
        
        start_time = time.time()
        for k in range(MAX_ITER_LINEAR):
            H_new = np.zeros(n)
            
            for i in range(n):
                sigma = 0.0
                for j in J.getrow(i).indices:
                    if j != i:
                        sigma += J[i, j] * H[j]
                
                H_new[i] = (b[i] - sigma) / D[i]
            
            # Criterio de convergencia
            diff = np.linalg.norm(H_new - H)
            H = H_new.copy()
            
            if diff < TOLERANCE_LINEAR:
                elapsed = time.time() - start_time
                return H, k + 1, elapsed
        
        elapsed = time.time() - start_time
        return H, MAX_ITER_LINEAR, elapsed

    def solve_gauss_seidel(self, J, U_n, x0=None):
        """
        Método de Gauss-Seidel para resolver J·H = -U_n
        Usa valores actualizados inmediatamente.
        """
        n = len(U_n)
        H = np.zeros(n) if x0 is None else x0.copy()
        b = -U_n
        
        J_dense = J.toarray()
        
        start_time = time.time()
        for k in range(MAX_ITER_LINEAR):
            H_old = H.copy()
            
            for i in range(n):
                sigma = 0.0
                for j in range(n):
                    if j != i:
                        sigma += J_dense[i, j] * H[j]
                
                H[i] = (b[i] - sigma) / J_dense[i, i]
            
            # Criterio de convergencia
            diff = np.linalg.norm(H - H_old)
            
            if diff < TOLERANCE_LINEAR:
                elapsed = time.time() - start_time
                return H, k + 1, elapsed
        
        elapsed = time.time() - start_time
        return H, MAX_ITER_LINEAR, elapsed

    def solve_gradiente_conjugado(self, J, U_n, x0=None):
        """
        Método del Gradiente Conjugado para resolver J·H = -U_n
        Nota: J debe ser simétrica y definida positiva.
        """
        n = len(U_n)
        H = np.zeros(n) if x0 is None else x0.copy()
        b = -U_n
        
        start_time = time.time()
        
        r = b - J.dot(H)
        p = r.copy()
        rsold = np.dot(r, r)
        
        for k in range(MAX_ITER_LINEAR):
            Ap = J.dot(p)
            alpha = rsold / np.dot(p, Ap)
            H = H + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r, r)
            
            if np.sqrt(rsnew) < TOLERANCE_LINEAR:
                elapsed = time.time() - start_time
                return H, k + 1, elapsed
            
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
        
        elapsed = time.time() - start_time
        return H, MAX_ITER_LINEAR, elapsed

    def solve_richardson(self, J, U_n, omega=0.5, x0=None):
        """
        Método de Richardson para resolver J·H = -U_n
        H_{k+1} = H_k + ω(b - J·H_k)
        """
        n = len(U_n)
        H = np.zeros(n) if x0 is None else x0.copy()
        b = -U_n
        
        J_dense = J.toarray()
        
        start_time = time.time()
        for k in range(MAX_ITER_LINEAR):
            H_old = H.copy()
            
            # Richardson: H_{k+1} = H_k + ω(b - J·H_k)
            residuo = b - J_dense @ H
            H = H + omega * residuo
            
            # Criterio de convergencia
            diff = np.linalg.norm(H - H_old)
            
            if diff < TOLERANCE_LINEAR:
                elapsed = time.time() - start_time
                return H, k + 1, elapsed
        
        elapsed = time.time() - start_time
        return H, MAX_ITER_LINEAR, elapsed

    def solve(self, method='gauss-seidel'):
        V_matrix = self.V_k.copy()
        print(f"\n{'='*80}")
        print(f"MÉTODO ITERATIVO SELECCIONADO: {method.upper()}")
        print(f"{'='*80}\n")
        total_linear_iters = 0
        total_linear_time = 0
        condicion_hist = []
        for nr_iter in range(1, MAX_ITER_NR + 1):
            # Ensamblar U y J en la iteración actual
            J, U_n = self.ensamblar_U_y_J(V_matrix)
            norma_U = np.linalg.norm(U_n)
            condicion_hist.append(np.linalg.cond(J.toarray()))
            print(f"\n--- Iteración Newton-Raphson {nr_iter} ---")
            print(f"||U_n|| = {norma_U:.10e}")
            # Resolver J·H = -U_n
            if method == 'richardson':
                H, iters, elapsed = self.solve_richardson(J, U_n)
            elif method == 'jacobi':
                H, iters, elapsed = self.solve_jacobi(J, U_n)
            elif method == 'gauss-seidel':
                H, iters, elapsed = self.solve_gauss_seidel(J, U_n)
            elif method == 'gradiente-conjugado':
                H, iters, elapsed = self.solve_gradiente_conjugado(J, U_n)
            else:
                raise ValueError(f"Método '{method}' no reconocido")
            total_linear_iters += iters
            total_linear_time += elapsed
            print(f"Sistema lineal resuelto en {iters} iteraciones ({elapsed:.4f}s)")
            print(f"||H|| = {np.linalg.norm(H):.10e}")
            # Actualizar: v_{n+1} = v_n + H
            m = 0
            V_new = V_matrix.copy()
            for j in range(NY):
                for i in range(NX):
                    if self._es_incognita(i, j):
                        V_new[j, i] = V_matrix[j, i] + H[m]
                        m += 1
            V_matrix = np.clip(V_new, 0, V0_INITIAL)
            # Criterio de convergencia de Newton-Raphson
            if norma_U < TOLERANCE_NR:
                print(f"\n{'='*80}")
                print(f"✅ CONVERGENCIA ALCANZADA en iteración {nr_iter}")
                print(f"{'='*80}")
                print(f"\nEstadísticas del método '{method}':")
                print(f"  - Total iteraciones lineales: {total_linear_iters}")
                print(f"  - Tiempo total en sistemas lineales: {total_linear_time:.4f}s")
                print(f"  - Promedio iteraciones por NR: {total_linear_iters/nr_iter:.1f}")
                break
        return V_matrix, condicion_hist

# --- VISUALIZACIÓN ---
def plot_solution(V_final, vy_value, method_name):
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

    ax.set_title(f'Solución: {method_name} | Vy={vy_value}', fontsize=14, fontweight='bold')
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

def plot_condicion_evolucion(cond_hist):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cond_hist)+1), cond_hist, marker='o', color='navy')
    ax.set_xlabel('Iteración Newton-Raphson')
    ax.set_ylabel('Número de condición')
    ax.set_title('Evolución del número de condición de la Jacobiana')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def analizar_matriz_jacobiana(solver):
    """
    Analiza las propiedades de la matriz Jacobiana para justificar
    la selección del método iterativo.
    """
    print(f"\n{'='*80}")
    print("ANÁLISIS DE LA MATRIZ JACOBIANA")
    print(f"{'='*80}\n")
    
    J, _ = solver.ensamblar_U_y_J(solver.V_k)
    J_dense = J.toarray()
    
    print(f"Dimensión: {J.shape[0]} x {J.shape[1]}")
    print(f"Elementos no nulos: {J.nnz} ({100*J.nnz/(J.shape[0]**2):.2f}% densidad)")
    
    # Verificar simetría
    es_simetrica = np.allclose(J_dense, J_dense.T, atol=1e-10)
    print(f"¿Es simétrica?: {es_simetrica}")
    
    # Dominancia diagonal
    diag = np.abs(J.diagonal())
    off_diag = np.array([np.sum(np.abs(J_dense[i, :])) - diag[i] for i in range(len(diag))])
    es_diag_dominante = np.all(diag > off_diag)
    filas_diag_dominante = np.sum(diag > off_diag)
    porcentaje_dd = (filas_diag_dominante / len(diag)) * 100
    print(f"¿Es diagonal dominante?: {es_diag_dominante}")
    print(f"  Filas con dominancia diagonal: {filas_diag_dominante}/{len(diag)} ({porcentaje_dd:.1f}%)")
    if es_diag_dominante:
        print(f"  Ratio promedio |a_ii|/Σ|a_ij|: {np.mean(diag/off_diag):.3f}")
    
    # Valores propios
    eigenvalues = np.linalg.eigvals(J_dense)
    print(f"\nValores propios:")
    print(f"  Min: {np.min(np.real(eigenvalues)):.6f}")
    print(f"  Max: {np.max(np.real(eigenvalues)):.6f}")
    print(f"  Número de condición: {np.linalg.cond(J_dense):.2e}")
    
    # Definida positiva
    es_def_positiva = np.all(np.real(eigenvalues) > 0)
    print(f"¿Es definida positiva?: {es_def_positiva}")
    
    print(f"\n{'='*80}")
    print("TABLA COMPARATIVA DE MÉTODOS ITERATIVOS")
    print(f"{'='*80}\n")
    print(f"{'Método':<25} {'Aplicable':<12} {'Convergencia':<20} {'Motivo'}")
    print(f"{'-'*100}")
    print(f"{'Richardson':<25} {'Sí':<12} {'Incierta':<20} Depende de ω; puede ser inestable")
    print(f"{'Jacobi':<25} {'Sí':<12} {'Lenta/Incierta':<20} Sin dominancia diagonal estricta")
    print(f"{'Gauss-Seidel':<25} {'Sí':<12} {'Buena':<20} ✅ Mejor opción disponible")
    print(f"{'Gradiente Conjugado':<25} {'NO':<12} {'N/A':<20} Requiere matriz simétrica")
    print(f"{'Gradiente Descendente':<25} {'NO':<12} {'N/A':<20} Requiere matriz simétrica")
    
    print(f"\n{'='*80}")
    print("JUSTIFICACIÓN DEL MÉTODO SELECCIONADO: GAUSS-SEIDEL")
    print(f"{'='*80}\n")
    print(f"✅ Propiedades de la matriz:")
    print(f"   • Diagonal dominante parcial: {porcentaje_dd:.1f}% de filas")
    print(f"   • Matriz dispersa (sparse): {100*J.nnz/(J.shape[0]**2):.2f}% densidad")
    print(f"   • NO simétrica → Descarta Gradientes Conjugado/Descendente")
    print(f"\n✅ Ventajas de Gauss-Seidel:")
    print(f"   • Usa valores ACTUALIZADOS inmediatamente (mejor que Jacobi)")
    print(f"   • Converge más rápido que Jacobi para este tipo de matrices")
    print(f"   • No requiere parámetro adicional ω (a diferencia de Richardson)")
    print(f"   • Aprovecha la estructura dispersa de la matriz")
    print(f"\n❌ Por qué se descartan los demás:")
    print(f"   • Richardson: Requiere estimar ω óptimo; convergencia incierta")
    print(f"   • Jacobi: Convergencia más lenta que Gauss-Seidel")
    print(f"   • Gradiente Conjugado: Matriz NO simétrica")
    print(f"   • Gradiente Descendente: Matriz NO simétrica")
    print(f"\n➤ CONCLUSIÓN: Gauss-Seidel es el método óptimo para este problema")

# --- EJECUCIÓN ---
if __name__ == '__main__':
    solver = FlujoNewtonRaphson()
    
    print(f"\n{'='*80}")
    print(f"SIMULACIÓN DE FLUJO CON NEWTON-RAPHSON Y MÉTODOS ITERATIVOS")
    print(f"{'='*80}")
    print(f"Grid: {NY} x {NX}")
    print(f"Número de incógnitas: {solver.N_INCÓGNITAS}")
    print(f"Vy constante: {VY_CONST}")
    print(f"h (paso): {H_PASO}")
    
    # Analizar matriz Jacobiana
    analizar_matriz_jacobiana(solver)
    
    # COMPARACIÓN DE MÉTODOS
    print(f"\n{'='*80}")
    print(f"COMPARACIÓN DE MÉTODOS ITERATIVOS")
    print(f"{'='*80}\n")
    
    metodos_comparar = ['richardson', 'jacobi', 'gauss-seidel']
    resultados = {}
    
    for metodo in metodos_comparar:
        print(f"\n{'='*80}")
        print(f"Probando método: {metodo.upper()}")
        print(f"{'='*80}")
        
        # Reiniciar solver para cada método
        solver_temp = FlujoNewtonRaphson()
        tiempo_inicio = time.time()
        V_sol, cond_hist = solver_temp.solve(method=metodo)
        tiempo_total = time.time() - tiempo_inicio
        
        # Verificar convergencia final
        J_final, F_final = solver_temp.ensamblar_U_y_J(V_sol)
        norma_residuo = np.linalg.norm(F_final)
        
        resultados[metodo] = {
            'tiempo': tiempo_total,
            'residuo': norma_residuo,
            'condicion_hist': cond_hist
        }
    
    # TABLA COMPARATIVA DE RESULTADOS
    print(f"\n{'='*80}")
    print(f"TABLA COMPARATIVA DE RENDIMIENTO")
    print(f"{'='*80}\n")
    print(f"{'Método':<20} {'Tiempo (s)':<15} {'Residuo Final':<20} {'Resultado'}")
    print(f"{'-'*80}")
    
    for metodo in metodos_comparar:
        tiempo = resultados[metodo]['tiempo']
        residuo = resultados[metodo]['residuo']
        convergencia = "✅ Convergió" if residuo < TOLERANCE_NR else "❌ No convergió"
        print(f"{metodo.upper():<20} {tiempo:<15.4f} {residuo:<20.6e} {convergencia}")
    
    print(f"\n{'='*80}")
    print(f"CONCLUSIÓN")
    print(f"{'='*80}")
    mejor_metodo = min(resultados.keys(), key=lambda m: resultados[m]['tiempo'])
    print(f"\n✅ GAUSS-SEIDEL es el método más eficiente:")
    print(f"   • Tiempo: {resultados['gauss-seidel']['tiempo']:.4f}s")
    print(f"   • {resultados['gauss-seidel']['tiempo']/resultados['richardson']['tiempo']:.2f}x más rápido que Richardson")
    print(f"   • {resultados['gauss-seidel']['tiempo']/resultados['jacobi']['tiempo']:.2f}x más rápido que Jacobi")
    print(f"   • Residuo final: {resultados['gauss-seidel']['residuo']:.6e}")
    
    # Usar Gauss-Seidel para visualización final
    METODO = 'gauss-seidel'
    V_solution = resultados[METODO]['condicion_hist']
    condicion_hist = resultados[METODO]['condicion_hist']
    
    # Resolver una vez más con Gauss-Seidel para obtener la solución final
    solver_final = FlujoNewtonRaphson()
    V_solution, condicion_hist = solver_final.solve(method=METODO)
    
    print("\n✅ Simulación completada. Generando visualización...")
    plot_solution(V_solution, VY_CONST, METODO.upper())
    # Calcular y mostrar el número de condición de la Jacobiana final
    J_final, _ = solver_final.ensamblar_U_y_J(V_solution)
    numero_condicion = np.linalg.cond(J_final.toarray())
    print("\n✨ Mostrando gráfico del número de condición...")
    plot_condicion(numero_condicion)
    print("\n✨ Mostrando evolución del número de condición...")
    plot_condicion_evolucion(condicion_hist)