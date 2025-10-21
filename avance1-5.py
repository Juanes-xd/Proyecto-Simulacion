import matplotlib.pyplot as plt
import numpy as np

# Dimensiones de la malla
alto = 40
ancho = 400

# Crear la malla de puntos
malla_x, malla_y = np.meshgrid(np.arange(1, ancho + 1), np.arange(1, alto + 1))

# Coordenadas de los 9 puntos de interés
puntos_interes = [(1, 1), (1, 20), (1, 39),
                  (200, 39), (200, 20), (200, 1),
                  (399, 39), (399, 20), (399, 1)]

# Crear el gráfico
plt.figure(figsize=(15, 6))

# Dibujar todos los puntos de la malla
plt.plot(malla_x, malla_y, 'o', color='lightgray', markersize=2)
plt.plot(malla_x.T, malla_y.T, 'o', color='lightgray', markersize=2)

# Resaltar los 9 puntos de interés
puntos_x = [p[0] for p in puntos_interes]
puntos_y = [p[1] for p in puntos_interes]
plt.plot(puntos_x, puntos_y, 'ro', markersize=8)

# Añadir etiquetas a los puntos para mayor claridad
etiquetas = ["(1,1)", "(1,20)", "(1,39)",
             "(200,39)", "(200,20)", "(200,1)",
             "(399,39)", "(399,20)", "(399,1)"]
for i, (x, y) in enumerate(puntos_interes):
    plt.text(x, y + 1, etiquetas[i], ha='center', fontsize=9, fontweight='bold')

# Configuración del gráfico
plt.title('Malla de 40x400 con Nodos Destacados')
plt.xlabel('Coordenada X (ancho)')
plt.ylabel('Coordenada Y (alto)')
plt.grid(True)
plt.show()