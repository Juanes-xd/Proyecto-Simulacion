import numpy as np
import matplotlib.pyplot as plt

# Tamaño de la malla
Nx, Ny = 400, 40   # 400 en x, 40 en y
hx, hy = 1, 1      # tamaño de paso (puede ajustarse)

# Creamos coordenadas
x = np.arange(0, Nx*hx, hx)
y = np.arange(0, Ny*hy, hy)
X, Y = np.meshgrid(x, y)

# Inicializamos dominio (1 = fluido, 0 = viga)
domain = np.ones((Ny, Nx))

# --- Definimos vigas (aproximadas) ---
# Viga inferior central (como en la figura)
domain[0:10, 180:220] = 0   # bloque en el centro inferior

# Viga superior derecha
domain[30:40, 340:380] = 0  # bloque en la parte superior derecha

# --- Definimos 9 puntos representativos ---
points = {
    "Esquina sup izq": (0, Ny-1),
    "Esquina sup der": (Nx-1, Ny-1),
    "Esquina inf izq": (0, 0),
    "Esquina inf der": (Nx-1, 0),
    "Centro sup": (Nx//2, Ny-1),
    "Centro inf": (Nx//2, 0),
    "Centro izq": (0, Ny//2),
    "Centro der": (Nx-1, Ny//2),
    "Centro abs": (Nx//2, Ny//2)
}

# --- Graficamos ---
plt.figure(figsize=(16, 4))
plt.imshow(domain, extent=[0, Nx, 0, Ny], origin='lower', cmap="Greys")

# Dibujar la malla con líneas
for i in range(0, Nx, 20):  # líneas verticales cada 20 pasos
    plt.axvline(i, color='lightgrey', linewidth=0.5)
for j in range(0, Ny, 2):   # líneas horizontales cada 2 pasos
    plt.axhline(j, color='lightgrey', linewidth=0.5)

# Dibujar nodos
plt.scatter(X, Y, s=1, color='blue', alpha=0.5)

# Dibujar los 9 puntos
for name, (px, py) in points.items():
    plt.scatter(px, py, color='red', s=40, label=name)

plt.title("Malla 40x400 con nodos, vigas y puntos representativos")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right", fontsize=7)
plt.show()
