# 📘 Simulación de Flujo con Método de Newton-Raphson

## 🎯 Objetivo del Proyecto

Este proyecto resuelve las **ecuaciones de Navier-Stokes** para simular el flujo de un fluido alrededor de obstáculos (vigas) en un dominio rectangular usando el **método de Newton-Raphson**.

---

## 📁 Archivo: `campo_velocidadesV2.py`

### **Descripción General**

El código calcula el **campo de velocidades horizontales (V_x)** en una malla rectangular de 5×50 celdas, donde:
- El fluido **entra** por la izquierda con velocidad V=1.0
- El fluido **sale** por la derecha con velocidad V=0.0
- Hay **dos vigas** (obstáculos) que obstruyen el flujo
- El **techo** tiene velocidad V=1.0 y el **piso** tiene velocidad V=0.0

---

## 🏗️ Estructura del Código

### 1️⃣ **Parámetros Globales**

```python
NY, NX = 5, 50      # Malla de 5 filas × 50 columnas
H = 8               # Tamaño de celda (referencia física)
VY_TEST = 0.1       # Componente de velocidad vertical (vorticidad)
MAX_ITER = 30       # Máximo de iteraciones de Newton-Raphson
TOLERANCE = 1e-8    # Criterio de convergencia
V0_INITIAL = 1.0    # Velocidad inicial/entrada
```

### 2️⃣ **Geometría del Dominio**

El dominio contiene dos vigas (obstáculos) con velocidad V=0:

**Viga Inferior:**
- Ubicación: Filas 0-1, Columnas 20-29
- Representa un obstáculo en la parte inferior central

**Viga Superior:**
- Ubicación: Fila 4, Columnas 40-49
- Representa un obstáculo en la parte superior derecha

```
Techo (V=1.0)     [==============================================]
                  [                        🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥]  ← Viga Superior
Flujo →           [                                              ]
                  [      🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥                       ]  ← Viga Inferior
Piso (V=0.0)      [==============================================]
              Entrada                                        Salida
              (V=1.0)                                        (V=0.0)
```

---

## 🔧 Clase `FlujoNewtonRaphson`

### **Inicialización**

```python
def __init__(self):
    1. Identifica qué nodos son incógnitas (interior, no frontera, no viga)
    2. Crea un mapa lineal de índices para el sistema matricial
    3. Inicializa la matriz de velocidades con condiciones de frontera
```

### **Método `_es_incognita(i, j)`**

Determina si el nodo (i, j) es una variable a resolver:
- ✅ Debe estar en el **interior** del dominio (`1 ≤ i ≤ NX-2`, `1 ≤ j ≤ NY-2`)
- ❌ No debe estar en una **viga**
- ❌ No debe estar en las **fronteras** (entrada, salida, techo, piso)

### **Método `_inicializar_matriz_velocidades()`**

Establece las **condiciones de frontera**:
- **V = 1.0**: Techo (fila 4) y Entrada (columna 0)
- **V = 0.0**: Piso (fila 0), Salida (columna 49), Vigas
- **Incógnitas**: Inicializadas con gradiente vertical proporcional

---

## 🧮 Ecuación Discretizada

### **Ecuación de Navier-Stokes Simplificada**

Para cada nodo central `(i, j)` con velocidad `V_c` y vecinos `V_r, V_l, V_u, V_d`:

```
F(V) = 4·V_c - (V_r + V_l + V_u + V_d) 
       + 4·V_c·(V_r - V_l) 
       + 4·VY_TEST·(V_u - V_d) = 0
```

Donde:
- `V_c` = Velocidad en el nodo central (i, j)
- `V_r` = Velocidad en el vecino derecho (i+1, j)
- `V_l` = Velocidad en el vecino izquierdo (i-1, j)
- `V_u` = Velocidad en el vecino superior (i, j+1)
- `V_d` = Velocidad en el vecino inferior (i, j-1)
- `VY_TEST` = Parámetro de convección vertical

---

## 🔄 Método de Newton-Raphson

### **Algoritmo Iterativo**

El método resuelve el sistema no lineal `F(V) = 0` mediante iteraciones:

```python
for k in range(MAX_ITER):
    1. Ensamblar Jacobiana J y lado derecho rhs = -F
    2. Calcular número de condición: κ(J)
    3. Resolver sistema lineal: J·ΔV = rhs
    4. Actualizar solución: V_new = V_old + 0.6·ΔV
    5. Aplicar límites: V ∈ [0, 1]
    6. Verificar convergencia: max(|ΔV|) < TOLERANCE
```

### **Matriz Jacobiana**

La Jacobiana contiene las derivadas parciales de F:

```python
∂F/∂V_c  = 4 + 4·(V_r - V_l)    # Diagonal principal
∂F/∂V_r  = -1 + 4·V_c           # Vecino derecho
∂F/∂V_l  = -1 - 4·V_c           # Vecino izquierdo
∂F/∂V_u  = -1 + 4·VY_TEST       # Vecino superior
∂F/∂V_d  = -1 - 4·VY_TEST       # Vecino inferior
```

### **Número de Condición**

El código calcula `κ(J) = ||J|| · ||J⁻¹||` en cada iteración:
- **κ(J) ≈ 1-10**: Matriz bien condicionada (estable)
- **κ(J) ≈ 10³-10⁶**: Razonablemente estable
- **κ(J) > 10⁸**: Mal condicionada (problemas numéricos)

---

## 📊 Visualización

### **Función `plot_solution(V_final, vy_value)`**

Genera un **mapa de calor** que muestra:

1. **Colores**: 
   - 🔵 Azul oscuro = Velocidad baja (≈0)
   - 🟡 Amarillo = Velocidad alta (≈1)

2. **Vigas**: Rectángulos rojos que representan los obstáculos

3. **Grid**: Líneas negras que delimitan cada celda de la malla

4. **Valores numéricos**: Velocidad exacta en cada celda (formato: 0.XX)

---

## ⚙️ Flujo de Ejecución

```
1. Crear instancia de FlujoNewtonRaphson
2. Inicializar matriz con condiciones de frontera
3. Bucle de Newton-Raphson:
   │
   ├─ Ensamblar sistema J·ΔV = -F
   ├─ Calcular κ(J)
   ├─ Resolver para ΔV
   ├─ Actualizar V (con amortiguamiento 0.6)
   └─ Verificar convergencia
4. Visualizar campo de velocidades
```

---

## 🎓 Conceptos Clave

### **Diferencias Finitas**
Las derivadas espaciales se aproximan usando valores en nodos vecinos:
```
∂V/∂x ≈ (V_r - V_l)/(2·h)
```

### **Newton-Raphson**
Método iterativo que lineariza la ecuación no lineal:
```
F(V + ΔV) ≈ F(V) + J·ΔV = 0
→ J·ΔV = -F
```

### **Matriz Sparse**
La Jacobiana tiene muy pocos elementos no-ceros (solo 5 por fila), por lo que se usa formato sparse para eficiencia.

### **Amortiguamiento**
El factor 0.6 en la actualización (`V_new = V_old + 0.6·ΔV`) previene oscilaciones y mejora la estabilidad.

---

## 📈 Resultados Esperados

Al ejecutar el código:
```bash
python campo_velocidadesV2.py
```

Se obtiene:
1. **Consola**: Progreso de iteraciones con cambio máximo y número de condición
2. **Ventana gráfica**: Campo de velocidades con:
   - Gradiente de velocidad de izquierda a derecha (alta → baja)
   - Perturbaciones alrededor de las vigas
   - Valores numéricos en cada celda

**Ejemplo de salida en consola:**
```
--- INICIANDO SIMULACIÓN ESTÁTICA (Vy=0.1) ---
Número total de incógnitas a resolver: 88
Iteración 1: Cambio máximo = 0.1234567890 | Número de condición = 1.23e+02
Iteración 2: Cambio máximo = 0.0456789012 | Número de condición = 1.45e+02
...
Iteración 8: Cambio máximo = 0.0000000005 | Número de condición = 1.67e+02
✅ Convergencia alcanzada en 8 iteraciones.
```

---

## 🔍 Interpretación Física

- **Velocidad alta (amarillo)**: Cerca de la entrada y en zonas sin obstáculos
- **Velocidad baja (azul)**: Cerca de la salida y alrededor de las vigas
- **Perturbaciones**: El flujo se acelera al pasar entre las vigas
- **Gradiente**: La velocidad decrece de izquierda a derecha debido a la pérdida de presión

---

## 📚 Referencias

- Método de Newton-Raphson para sistemas no lineales
- Ecuaciones de Navier-Stokes (forma simplificada 2D)
- Diferencias finitas en mallas estructuradas
- Método de relajación sucesiva (SOR)

---

## 👨‍💻 Autor

Proyecto de Simulación Computacional
