import numpy as np
import matplotlib.pyplot as plt

# Dimensiones del dominio
Lx, Ly = 4.0, 0.4
Nx, Ny = 400, 40
hx, hy = Lx/(Nx-1), Ly/(Ny-1)

# Coordenadas
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y, indexing="ij")

# Subdividimos en 9 cuadros (3x3)
div_x = [0, Lx/3, 2*Lx/3, Lx]
div_y = [0, Ly/3, 2*Ly/3, Ly]

# Puntos representativos (centros de cada cuadro 3x3)
rep_points = []
for i in range(3):
    for j in range(3):
        cx = 0.5*(div_x[i]+div_x[i+1])
        cy = 0.5*(div_y[j]+div_y[j+1])
        rep_points.append((cx, cy))

# Graficamos
plt.figure(figsize=(12,3))
plt.pcolormesh(X, Y, np.ones_like(X), shading="auto", cmap="Blues", alpha=0.1)

# Dibujar las divisiones de 9 cuadros
for dx in div_x:
    plt.axvline(dx, color="k", linestyle="--")
for dy in div_y:
    plt.axhline(dy, color="k", linestyle="--")

# Marcar puntos representativos
for (cx, cy) in rep_points:
    plt.plot(cx, cy, "ro")
    plt.text(cx, cy+0.01, f"({cx:.2f},{cy:.2f})", ha="center", fontsize=8)

plt.title("Malla 40x400 subdividida en 9 cuadros con puntos representativos")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
