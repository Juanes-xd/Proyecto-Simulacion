import numpy as np
import matplotlib.pyplot as plt

# Tamaño físico del dominio
Lx, Ly = 400, 40   # ancho y alto

# Número de divisiones (menos que 400x40)
Nx, Ny = 80, 10    # nodos en x e y

# Espaciamiento
hx, hy = Lx/(Nx-1), Ly/(Ny-1)

# Coordenadas de la malla
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Máscara de dominio (1 = fluido, 0 = viga)
domain = np.ones((Ny, Nx))

# --- Definimos vigas (aprox. escaladas al tamaño reducido) ---
# Viga inferior central
ix1, ix2 = int(180/Lx*(Nx-1)), int(220/Lx*(Nx-1))
jy1, jy2 = 0, int(10/Ly*(Ny-1))   # altura ~10
domain[jy1:jy2+1, ix1:ix2+1] = 0

# Viga superior derecha
# ...existing code...

# Viga superior derecha (pegada a la pared derecha)
ix1 = int(360/Lx*(Nx-1))  # Ajusta el inicio más a la derecha
ix2 = Nx-1                # Hasta el borde derecho
jy1 = int(30/Ly*(Ny-1))
jy2 = Ny-1
domain[jy1:jy2+1, ix1:ix2+1] = 0

# ...existing code...

# --- Puntos representativos (aprox en la malla reducida) ---
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
plt.figure(figsize=(14, 3))
plt.imshow(domain, extent=[0, Lx, 0, Ly], origin='lower', cmap="Greys")

# Dibujar la malla
for i in range(Nx):
    plt.axvline(i*hx, color='lightgrey', linewidth=0.5)
for j in range(Ny):
    plt.axhline(j*hy, color='lightgrey', linewidth=0.5)

# Dibujar nodos
plt.scatter(X, Y, s=10, color='blue', alpha=0.6)

# Dibujar puntos representativos
for name, (px, py) in points.items():
    plt.scatter(x[px], y[py], color='red', s=50, label=name)

plt.title(f"Malla reducida {Ny}x{Nx} (dominio 40x400)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper right", fontsize=7)
plt.show()
