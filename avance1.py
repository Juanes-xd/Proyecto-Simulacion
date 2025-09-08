import numpy as np
import matplotlib.pyplot as plt

# Tamaño del dominio
Lx, Ly = 4.0, 0.4    # largo x y altura y (ejemplo en metros)
Nx, Ny = 400, 40     # puntos en x e y
hx, hy = Lx/(Nx-1), Ly/(Ny-1)

# Malla (coordenadas)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Inicialización de campos
u = np.zeros((Nx, Ny))  # velocidad en x
v = np.zeros((Nx, Ny))  # velocidad en y
p = np.zeros((Nx, Ny))  # presión

# Máscara para obstáculos (1=fluido, 0=sólido)
mask = np.ones((Nx, Ny))

# Ejemplo: obstáculo central (viga en el piso)
# Supongamos ocupa de x=1.6 a x=2.0, y=0 a y=0.15
for i in range(Nx):
    for j in range(Ny):
        if (1.6 <= X[i,j] <= 2.0) and (0.0 <= Y[i,j] <= 0.15):
            mask[i,j] = 0

# Otra viga (arriba a la derecha)
for i in range(Nx):
    for j in range(Ny):
        if (3.3 <= X[i,j] <= 3.7) and (0.25 <= Y[i,j] <= 0.40):
            mask[i,j] = 0

# Visualización de la malla y obstáculos
plt.figure(figsize=(10,2))
plt.pcolormesh(X, Y, mask, shading='auto', cmap='gray_r')
plt.title("Malla 40x400 con obstáculos")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="1=fluido, 0=obstáculo")
plt.show()
