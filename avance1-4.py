import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Dimensiones del dominio
Lx, Ly = 400, 40  

# Tamaño de celda cuadrada
dx = dy = 2  

# Número de celdas
Nx_cells, Ny_cells = int(Lx/dx), int(Ly/dy)

# Crear figura
fig, ax = plt.subplots(figsize=(14, 3))

# Dibujar las celdas como rectángulos
for i in range(Nx_cells):
    for j in range(Ny_cells):
        rect = patches.Rectangle((i*dx, j*dy), dx, dy,
                                 linewidth=0.3, edgecolor="black",
                                 facecolor="none")
        ax.add_patch(rect)

# Añadir vigas (ajustadas a la imagen de referencia)
# Viga C (central inferior)
vigaC = patches.Rectangle((160, 0), 80, 20,
                          linewidth=1, edgecolor="black", facecolor="grey")
# Viga J (superior derecha)
vigaJ = patches.Rectangle((320, 30), 80, 10,
                          linewidth=1, edgecolor="black", facecolor="grey")

ax.add_patch(vigaC)
ax.add_patch(vigaJ)

# Etiquetas y estilo
ax.set_title(f"Malla cuadrada {Nx_cells}x{Ny_cells} = {Nx_cells*Ny_cells} celdas\nΔx=Δy={dx}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_aspect("equal")

plt.show()
