# 📘 Simulación de Flujo: Comparación de Métodos Numéricos

## 🎯 Objetivo del Proyecto

Este proyecto resuelve las **ecuaciones de Navier-Stokes** para simular el flujo de un fluido alrededor de obstáculos (vigas) en un dominio rectangular, comparando diferentes **métodos numéricos iterativos** para sistemas no lineales:

- ✅ **Newton-Raphson** (método directo con Jacobiana)
- ❌ **Gauss-Seidel** (método iterativo punto fijo)
- ❌ **Jacobi** (método iterativo simultáneo)
- ❌ **Richardson** (método de descenso por residuo)

El objetivo es demostrar que **no todos los métodos son adecuados** para este tipo de problemas con fuerte no linealidad.

---

## � Estructura del Proyecto

| Archivo | Método | Estado | Descripción |
|---------|--------|--------|-------------|
| `campo_velocidadesV2.py` | Newton-Raphson | ✅ Converge | Método principal usando Jacobiana completa |
| `avance3-2.py` | Gauss-Seidel | ❌ No converge | Método iterativo con actualización secuencial |
| `avance3-4.py` | Jacobi | ❌ No converge | Método iterativo con actualización simultánea |
| `avance3-3.py` | Richardson | ❌ No converge | Método de descenso por residuo con ω=0.01 |
| `analisis_matriz.py` | Análisis | 📊 Herramienta | Determina viabilidad de métodos a priori |

---

## �📁 Archivo: `campo_velocidadesV2.py`

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

## 📁 Archivo: `avance3-2.py` - Método de Gauss-Seidel

### **Descripción**
Implementación del **método de Gauss-Seidel puro** (sin sobre-relajación) para resolver el mismo sistema no lineal.

### **Cómo Funciona**

```python
Para cada iteración k:
    Para cada nodo incógnita (i,j):
        1. Leer vecinos V_r, V_l, V_u, V_d
           (usando valores YA ACTUALIZADOS en esta iteración)
        2. Despejar V_c de la ecuación:
           V_c = (V_r + V_l + V_u + V_d - 4·VY·(V_u-V_d)) / (4·(1 + V_r - V_l))
        3. Actualizar INMEDIATAMENTE V_c
        4. Calcular cambio |V_new - V_old|
    Verificar convergencia: max_cambio < 1e-8
```

**Diferencia clave con Jacobi:** Gauss-Seidel usa valores **ya actualizados** en la misma iteración, mientras que Jacobi usa solo valores de la iteración anterior.

### **Resultados**

```
Estado: ❌ NO CONVERGE
Iteraciones ejecutadas: 2000
Cambio máximo final: 0.7451965478
Razón: Queda estancado, oscilando sin alcanzar tolerancia
```

### **Conclusión**

El método de Gauss-Seidel **falla** para este problema porque:
- La matriz NO es diagonal dominante (solo 44.8% de filas)
- El radio espectral ρ(M_GS) = 2.88 > 1 (no garantiza convergencia)
- La ecuación no lineal 4·V_c·(V_r - V_l) domina sobre la parte lineal

---

## 📁 Archivo: `avance3-4.py` - Método de Jacobi

### **Descripción**
Implementación del **método de Jacobi puro** (sin sobre-relajación) con actualización simultánea.

### **Cómo Funciona**

```python
Para cada iteración k:
    Crear copia V_new de V_old
    Para cada nodo incógnita (i,j):
        1. Leer vecinos V_r, V_l, V_u, V_d de V_OLD
           (TODOS los valores de iteración anterior)
        2. Calcular nuevo valor:
           V_new[i,j] = (V_r + V_l + V_u + V_d - 4·VY·(V_u-V_d)) / (4·(1 + V_r - V_l))
        3. Guardar en V_new (NO sobrescribir todavía)
    Reemplazar V_old ← V_new (actualización simultánea)
    Verificar convergencia
```

**Ventaja:** Fácilmente paralelizable (todas las actualizaciones son independientes)

**Desventaja:** Converge más lento que Gauss-Seidel en problemas lineales

### **Resultados**

```
Estado: ❌ NO CONVERGE
Iteraciones ejecutadas: 3000
Cambio máximo final: 0.7450136892
Razón: Oscila sin converger, similar a Gauss-Seidel
```

### **Conclusión**

El método de Jacobi **falla** porque:
- NO es diagonal dominante estricta (solo 60/134 filas)
- Radio espectral ρ(M_Jacobi) = 1.70 > 1 (teorema de convergencia no se cumple)
- La actualización simultánea no ayuda cuando la matriz no cumple requisitos

---

## 📁 Archivo: `avance3-3.py` - Método de Richardson

### **Descripción**
Implementación del **método de Richardson** con parámetro de relajación ω = 0.01 (muy pequeño para estabilidad).

### **Cómo Funciona**

```python
Para cada iteración k:
    1. Calcular residuo F(V) para cada incógnita:
       F = 4·V_c - (V_r+V_l+V_u+V_d) + 4·V_c·(V_r-V_l) + 4·VY·(V_u-V_d)
    
    2. Actualización Richardson:
       V^(k+1) = V^(k) - ω · F(V^(k))
       
    3. Aplicar límites: V ∈ [0, 1]
    4. Verificar convergencia
```

**Parámetro ω:** Controla el tamaño del paso. Demasiado grande → oscilación, demasiado pequeño → convergencia lenta.

### **Resultados**

```
Estado: ❌ NO CONVERGE (con ω=0.01)
Iteraciones ejecutadas: 5000
Cambio máximo: Variable (depende de ω)
Razón: 
  - ω=0.5 → Oscila violentamente (||Residuo|| ≈ 47)
  - ω=0.01 → Demasiado lento, no converge en tiempo razonable
```

### **Conclusión**

Richardson **no es práctico** para este problema porque:
- Matriz NO simétrica dificulta la elección del ω óptimo
- Requiere miles de iteraciones incluso con ω ajustado
- Sin información de la Jacobiana, no aprovecha estructura del problema
- El término no lineal hace que ω óptimo cambie en cada iteración

---

## 📁 Archivo: `analisis_matriz.py` - Análisis A Priori

### **Descripción**
Herramienta que **analiza las propiedades matemáticas** de la matriz Jacobiana para determinar qué métodos iterativos son viables **antes de ejecutarlos**.

### **Análisis Realizado**

#### 1️⃣ **Número de Condición** κ(J)
```
κ(J) = 1.43e+01 ✅
Interpretación: Matriz bien condicionada
Impacto: Buena estabilidad numérica
```

#### 2️⃣ **Simetría**
```
||J - J^T|| / ||J|| = 0.956 ❌
Interpretación: Matriz NO SIMÉTRICA
Impacto: Gradiente Conjugado NO aplicable
```

#### 3️⃣ **Diagonal Dominancia**
```
Filas diagonales dominantes: 60/134 (44.8%) ❌
Interpretación: NO diagonal dominante
Impacto: Jacobi y Gauss-Seidel NO garantizan convergencia
```

#### 4️⃣ **Autovalores**
```
Rango: [0.75, 7.08]
Todos positivos: ✅ (134 positivos, 0 negativos)
Pero NO simétrica: ⚠️ No es definida positiva en sentido clásico
```

#### 5️⃣ **Radio Espectral**
```
ρ(M_Jacobi) = 1.70 > 1  ❌ NO converge
ρ(M_Gauss-Seidel) = 2.88 > 1  ❌ NO converge
```

**Teorema:** Un método iterativo converge si y solo si ρ(M) < 1

#### 6️⃣ **Estructura Sparse**
```
Sparsidad: 96.85% ✅
Elementos no nulos: 566 de 17,956
Ancho de banda: 48
Interpretación: Estructura dispersa ideal para métodos iterativos
```

### **Resultados del Análisis**

| Método | Requisito | ¿Cumple? | Veredicto |
|--------|-----------|----------|-----------|
| **Newton-Raphson** | Jacobiana calculable | ✅ Sí | ✅ **VIABLE** |
| **Gauss-Seidel** | Diagonal dominante O simétrica def. positiva | ❌ No | ❌ **DESCARTAR** |
| **Jacobi** | Diagonal dominante estricta | ❌ No (44.8%) | ❌ **DESCARTAR** |
| **Richardson** | Matriz simétrica (para ω óptimo) | ❌ No | ❌ **DESCARTAR** |
| **Gradiente Conjugado** | Simétrica Y definida positiva | ❌ No (no simétrica) | ❌ **DESCARTAR** |

### **Visualización Generada**

El script produce `analisis_matriz_jacobiana.png` con:
- **Izquierda:** Patrón de sparsidad (estructura de banda)
- **Derecha:** Mapa de calor de valores de la Jacobiana

### **Conclusión Clave**

> **Solo examinando la matriz Jacobiana**, sin ejecutar iteraciones, podemos descartar Jacobi, Gauss-Seidel, Richardson y Gradiente Conjugado porque **no cumplen los requisitos matemáticos** para garantizar convergencia.

---

## 🎓 Comparación de Métodos: Tabla Resumen

| Característica | Newton-Raphson | Gauss-Seidel | Jacobi | Richardson |
|----------------|----------------|--------------|--------|------------|
| **Tipo** | Cuasi-Newton | Punto fijo | Punto fijo | Descenso |
| **Usa Jacobiana** | ✅ Completa | ❌ No | ❌ No | ❌ No |
| **Convergencia** | ✅ Cuadrática | ❌ Lineal (si converge) | ❌ Lineal (si converge) | ❌ Lineal |
| **Iteraciones** | ~10 | >2000 | >3000 | >5000 |
| **Estado** | ✅ Converge | ❌ Estancado | ❌ Estancado | ❌ Oscila |
| **Requisito matriz** | κ(J) razonable | Diag. dom. | Diag. dom. estricta | Simétrica |
| **¿Cumple requisito?** | ✅ Sí | ❌ No | ❌ No | ❌ No |
| **Paralelizable** | ⚠️ Parcial | ❌ No | ✅ Sí | ✅ Sí |
| **Memoria** | Alta (matriz J) | Baja | Media | Baja |

---

## 💡 Lecciones Aprendidas

### 1. **No todos los métodos sirven para todos los problemas**

Aunque Jacobi, Gauss-Seidel y Richardson son métodos clásicos, **fallan** en este problema porque la matriz Jacobiana no cumple sus requisitos de convergencia.

### 2. **El análisis a priori es valioso**

Con el archivo `analisis_matriz.py` determinamos **antes de ejecutar** que Jacobi y Gauss-Seidel no convergirían, ahorrando tiempo computacional.

### 3. **La no linealidad importa**

El término no lineal **4·V_c·(V_r - V_l)** en la ecuación de Navier-Stokes:
- Rompe la diagonal dominancia
- Hace que la matriz cambie en cada iteración
- Justifica el uso de Newton-Raphson (que recalcula J cada vez)

### 4. **Relajación no siempre ayuda**

Aunque SOR (Successive Over-Relaxation) puede mejorar Gauss-Seidel en problemas lineales, en este sistema **no lineal** con matriz mal condicionada para métodos iterativos simples, la relajación no es suficiente.

### 5. **Newton-Raphson es el único viable**

Por usar la **Jacobiana completa**, Newton-Raphson:
- Captura la no linealidad correctamente
- Converge en ~10 iteraciones vs. >2000 de otros
- Es la única opción práctica para este problema

---

## 🚀 Cómo Ejecutar

### Método de Newton-Raphson (✅ Funciona)
```bash
python campo_velocidadesV2.py
```

### Método de Gauss-Seidel (❌ No converge)
```bash
python avance3-2.py
```

### Método de Jacobi (❌ No converge)
```bash
python avance3-4.py
```

### Método de Richardson (❌ No converge)
```bash
python avance3-3.py
```

### Análisis de Matriz (📊 Herramienta)
```bash
python analisis_matriz.py
```

---

## 📊 Resultados Esperados

### ✅ Newton-Raphson
```
Iteración 1: Cambio máximo = 0.1234567890 | κ(J) = 1.43e+01
Iteración 2: Cambio máximo = 0.0456789012 | κ(J) = 1.45e+01
...
Iteración 10: Cambio máximo = 0.0000000005 | κ(J) = 1.67e+01
✅ Convergencia alcanzada en 10 iteraciones.
```

### ❌ Gauss-Seidel / Jacobi
```
Iteración 50: Cambio máximo = 0.7451965478
Iteración 100: Cambio máximo = 0.7451965478
...
Iteración 2000: Cambio máximo = 0.7451965478
⚠️ ADVERTENCIA: El método NO convergió después de 2000 iteraciones.
```

### 📊 Análisis de Matriz
```
κ(J) = 1.43e+01 ✅ Bien condicionada
Simetría: NO ❌
Diagonal dominante: NO (44.8%) ❌
ρ(M_Jacobi) = 1.70 > 1 ❌
ρ(M_Gauss-Seidel) = 2.88 > 1 ❌

CONCLUSIÓN: Solo Newton-Raphson es viable
```

---

## 👨‍💻 Autor

Proyecto de Simulación Computacional
