# -*- coding: utf-8 -*-
"""
TAREA 2 - MISCELÁNEA DE EJERCICIOS
Fundamentos de Análisis Cuantitativo - UNAD
Estudiante: Adrián Saldaña Pacheco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*60)
print("TAREA 2 - MISCELÁNEA DE EJERCICIOS")
print("="*60)

# ============================================================
# EJERCICIO 1: OPERACIONES CON MATRICES
# ============================================================
print("\n--- Ejercicio 1: Operaciones con matrices ---")
A = np.array([[2, 3], [1, 4]])
B = np.array([[1, 2], [3, 1]])
print("A + B =\n", A + B)
print("2A - 3B =\n", 2*A - 3*B)
print("A x B =\n", np.dot(A, B))
det_A = np.linalg.det(A)
print(f"Determinante de A: {det_A:.2f}")
print("¿Es invertible?", "Sí" if det_A != 0 else "No")

# ============================================================
# EJERCICIO 2: SISTEMA DE ECUACIONES (CRAMER)
# ============================================================
print("\n--- Ejercicio 2: Sistema de ecuaciones (Cramer) ---")
D = np.array([[2, 1, -1], [3, -2, 1], [1, 4, -3]])
b = np.array([5, -4, 3])
det_D = np.linalg.det(D)
print(f"Determinante del sistema: {det_D:.2f}")
if abs(det_D) < 1e-10:
    print("El determinante es cero. El sistema no tiene solución única.")
    print("Es indeterminado o inconsistente. No se puede aplicar Cramer.")
else:
    Dx = D.copy(); Dx[:,0] = b
    Dy = D.copy(); Dy[:,1] = b
    Dz = D.copy(); Dz[:,2] = b
    x = np.linalg.det(Dx)/det_D
    y = np.linalg.det(Dy)/det_D
    z = np.linalg.det(Dz)/det_D
    print(f"Solución: x={x:.2f}, y={y:.2f}, z={z:.2f}")

# ============================================================
# EJERCICIO 3: VALORES Y VECTORES PROPIOS
# ============================================================
print("\n--- Ejercicio 3: Valores y vectores propios ---")
C = np.array([[4, 1], [2, 3]])
valores, vectores = np.linalg.eig(C)
print("Valores propios:", valores)
print("Vectores propios (columnas):\n", vectores)

# ============================================================
# EJERCICIO 4: FUNCIÓN LINEAL f(x)=2x+3
# ============================================================
print("\n--- Ejercicio 4: Función lineal f(x)=2x+3 ---")
f = lambda x: 2*x + 3
print(f"f(2)={f(2)}, f(-1)={f(-1)}, f(0)={f(0)}")
x_vals = np.linspace(-2, 4, 100)
plt.figure()
plt.plot(x_vals, f(x_vals), label='f(x)=2x+3')
plt.scatter([2, -1, 0], [f(2), f(-1), f(0)], color='red', zorder=5)
plt.title('Ejercicio 4: Función lineal')
plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(); plt.grid(True)
plt.savefig('ej4_lineal.png'); plt.close()
print("Gráfica guardada: ej4_lineal.png")

# ============================================================
# EJERCICIO 5: FUNCIÓN CUADRÁTICA g(x)=x^2-4x+3
# ============================================================
print("\n--- Ejercicio 5: Función cuadrática ---")
g = lambda x: x**2 - 4*x + 3
raices = np.roots([1, -4, 3])
print("Raíces:", raices)
vertice_x = -(-4)/(2*1)
vertice_y = g(vertice_x)
print(f"Vértice: ({vertice_x}, {vertice_y})")
x_vals = np.linspace(-1, 5, 100)
plt.figure()
plt.plot(x_vals, g(x_vals), label='g(x)=x²-4x+3')
plt.scatter(raices, [0,0], color='red', label='Raíces')
plt.scatter([vertice_x], [vertice_y], color='green', marker='*', s=150, label='Vértice')
plt.title('Ejercicio 5: Función cuadrática')
plt.xlabel('x'); plt.ylabel('g(x)'); plt.legend(); plt.grid(True)
plt.savefig('ej5_cuadratica.png'); plt.close()
print("Gráfica guardada: ej5_cuadratica.png")

# ============================================================
# EJERCICIO 6: ECUACIÓN NO LINEAL x^3-6x^2+11x-6=0
# ============================================================
print("\n--- Ejercicio 6: Ecuación no lineal ---")
coef = [1, -6, 11, -6]
soluciones = np.roots(coef)
print("Soluciones:", soluciones)

# ============================================================
# EJERCICIOS 7 Y 8: ESTADÍSTICA DESCRIPTIVA
# ============================================================
print("\n--- Ejercicios 7 y 8: Estadística descriptiva ---")
edades = np.array([25, 30, 29, 32, 25, 40, 28, 33, 35, 29])
media = np.mean(edades)
mediana = np.median(edades)
moda = stats.mode(edades, keepdims=True)
print(f"Media: {media:.2f}")
print(f"Mediana: {mediana:.2f}")
print(f"Moda: {moda.mode[0]} (frecuencia {moda.count[0]})")
print(f"Rango: {np.ptp(edades)}")
print(f"Varianza muestral: {np.var(edades, ddof=1):.2f}")
print(f"Desviación estándar muestral: {np.std(edades, ddof=1):.2f}")

# ============================================================
# EJERCICIO 9: VISUALIZACIÓN DE VENTAS
# ============================================================
print("\n--- Ejercicio 9: Visualización de ventas ---")
ventas = np.array([200, 250, 300, 350, 400, 450])
meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(meses, ventas, color='skyblue')
plt.title('Gráfico de barras - Ventas mensuales')
plt.xlabel('Mes'); plt.ylabel('Ventas')
plt.subplot(1, 2, 2)
plt.hist(ventas, bins=5, color='lightcoral', edgecolor='black')
plt.title('Histograma - Distribución de ventas')
plt.xlabel('Ventas'); plt.ylabel('Frecuencia')
plt.tight_layout()
plt.savefig('ej9_ventas.png'); plt.close()
print("Gráfica guardada: ej9_ventas.png")

# ============================================================
# EJERCICIO 10: CLASIFICACIÓN DE CLIENTES
# ============================================================
print("\n--- Ejercicio 10: Clasificación de clientes ---")
clientes = np.array([[500, 200], [300, 100], [700, 400]])
media_clientes = np.mean(clientes, axis=0)
print(f"Media de gastos (alimentación, ocio): {media_clientes}")
for i, c in enumerate(clientes):
    grupo = "Alto" if c[0] > media_clientes[0] and c[1] > media_clientes[1] else "Bajo/Medio"
    print(f"Cliente {chr(65+i)}: {c} -> Grupo {grupo}")
plt.figure()
plt.scatter(clientes[:,0], clientes[:,1], s=100)
for i, c in enumerate(clientes):
    plt.annotate(chr(65+i), (c[0], c[1]), xytext=(5,5), textcoords='offset points')
plt.axhline(media_clientes[1], color='gray', linestyle='--')
plt.axvline(media_clientes[0], color='gray', linestyle='--')
plt.title('Ejercicio 10: Clasificación de clientes')
plt.xlabel('Gasto Alimentación'); plt.ylabel('Gasto Ocio')
plt.grid(True)
plt.savefig('ej10_clientes.png'); plt.close()
print("Gráfica guardada: ej10_clientes.png")

# ============================================================
# EJERCICIO 11: OPTIMIZACIÓN DE INVENTARIOS
# ============================================================
print("\n--- Ejercicio 11: Optimización de inventarios ---")
A_vals = np.linspace(0, 150, 400)
B1 = (5000 - 50*A_vals)/30
B2 = 150 - A_vals
plt.figure()
plt.plot(A_vals, B1, label='50A + 30B = 5000')
plt.plot(A_vals, B2, label='A + B = 150')
plt.fill_between(A_vals, 0, np.minimum(B1, B2), where=(np.minimum(B1,B2)>=0), alpha=0.3, color='green')
plt.xlim(0, 150); plt.ylim(0, 180)
plt.xlabel('Producto A'); plt.ylabel('Producto B')
plt.title('Ejercicio 11: Región factible')
plt.legend(); plt.grid(True)
plt.savefig('ej11_inventario.png'); plt.close()
print("Gráfica guardada: ej11_inventario.png")
print("Vértices: (0,0), (100,0), (0,150), (25,125)")

# ============================================================
# EJERCICIO 12: MEDIA MÓVIL
# ============================================================
print("\n--- Ejercicio 12: Media móvil de 3 días ---")
visitas = np.array([500, 520, 550, 530, 600, 620, 610, 650, 680, 700])
media_movil = pd.Series(visitas).rolling(window=3).mean()
print("Media móvil (3 días):")
print(media_movil.dropna().values)
plt.figure()
plt.plot(range(1, 11), visitas, marker='o', label='Serie original')
plt.plot(range(1, 11), media_movil, marker='s', label='Media móvil 3 días')
plt.title('Ejercicio 12: Visitas diarias y media móvil')
plt.xlabel('Día'); plt.ylabel('Visitas'); plt.legend(); plt.grid(True)
plt.savefig('ej12_media_movil.png'); plt.close()
print("Gráfica guardada: ej12_media_movil.png")

# ============================================================
# EJERCICIO 13: K-MEANS
# ============================================================
print("\n--- Ejercicio 13: K-means (2 grupos) ---")
datos_kmeans = np.array([[1000, 500], [1500, 800], [800, 300], [1200, 700], [500, 200]])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(datos_kmeans)
etiquetas = kmeans.labels_
centroides = kmeans.cluster_centers_
print("Etiquetas de grupo:", etiquetas)
print("Centroides:\n", centroides)
plt.figure()
plt.scatter(datos_kmeans[:,0], datos_kmeans[:,1], c=etiquetas, cmap='viridis', s=100)
plt.scatter(centroides[:,0], centroides[:,1], c='red', marker='X', s=200, label='Centroides')
for i, d in enumerate(datos_kmeans):
    plt.annotate(f'C{i+1}', (d[0], d[1]), xytext=(5,5), textcoords='offset points')
plt.title('Ejercicio 13: Segmentación K-means')
plt.xlabel('Gasto anual categoría 1'); plt.ylabel('Gasto anual categoría 2')
plt.legend(); plt.grid(True)
plt.savefig('ej13_kmeans.png'); plt.close()
print("Gráfica guardada: ej13_kmeans.png")

# ============================================================
# EJERCICIO 14: REGRESIÓN LINEAL
# ============================================================
print("\n--- Ejercicio 14: Regresión lineal ---")
proyectos = np.array([1, 2, 3, 4, 5, 6])
ingresos = np.array([50, 60, 70, 80, 90, 100])
m, b = np.polyfit(proyectos, ingresos, 1)
print(f"Ecuación: y = {m:.2f}x + {b:.2f}")
prediccion = m*8 + b
print(f"Predicción para 8 proyectos: {prediccion:.2f} mil dólares")
plt.figure()
plt.scatter(proyectos, ingresos, color='blue', label='Datos reales')
plt.plot(proyectos, m*proyectos + b, color='red', label=f'Regresión: y={m:.1f}x+{b:.1f}')
plt.scatter(8, prediccion, color='green', marker='*', s=150, label='Predicción (8)')
plt.title('Ejercicio 14: Regresión lineal')
plt.xlabel('Proyectos'); plt.ylabel('Ingresos (miles $)'); plt.legend(); plt.grid(True)
plt.savefig('ej14_regresion.png'); plt.close()
print("Gráfica guardada: ej14_regresion.png")

# ============================================================
# EJERCICIO 15: CORRELACIÓN DE PEARSON
# ============================================================
print("\n--- Ejercicio 15: Correlación de Pearson ---")
inversion = np.array([10, 20, 30, 40, 50])
ventas_corr = np.array([100, 150, 200, 250, 300])
r, p = stats.pearsonr(inversion, ventas_corr)
print(f"Coeficiente de Pearson: {r:.4f}")
print("Interpretación: Correlación fuerte y positiva perfecta." if r > 0.9 else "Correlación moderada/débil.")

# ============================================================
# EJERCICIO 16: INVENTARIO POR CATEGORÍA
# ============================================================
print("\n--- Ejercicio 16: Inventario por categoría ---")
electronicos = np.array([50, 60, 70, 80, 90])
ropa = np.array([100, 90, 110, 120, 100])
alimentos = np.array([200, 250, 300, 220, 270])
categorias = {'Electrónicos': electronicos, 'Ropa': ropa, 'Alimentos': alimentos}
for nombre, datos in categorias.items():
    media_cat = np.mean(datos)
    desv_cat = np.std(datos, ddof=1)
    cv = (desv_cat / media_cat) * 100
    print(f"{nombre}: Media={media_cat:.1f}, Desv={desv_cat:.2f}, CV={cv:.1f}%")
print("Interpretación: Ropa tiene el CV más bajo (más consistente).")

# ============================================================
# EJERCICIO 17: CALIFICACIONES
# ============================================================
print("\n--- Ejercicio 17: Calificaciones ---")
calificaciones = np.array([85, 90, 88, 92, 87, 91, 89, 93])
print(f"Media: {np.mean(calificaciones):.2f}")
print(f"Varianza: {np.var(calificaciones, ddof=1):.2f}")
print(f"Desviación estándar: {np.std(calificaciones, ddof=1):.2f}")

# ============================================================
# EJERCICIO 18: COMPARACIÓN DE EQUIPOS
# ============================================================
print("\n--- Ejercicio 18: Comparación de equipos ---")
equipo_A = np.array([75, 80, 85, 70, 90])
equipo_B = np.array([78, 82, 76, 85, 89])
print(f"Equipo A: Varianza={np.var(equipo_A, ddof=1):.2f}, Desv={np.std(equipo_A, ddof=1):.2f}")
print(f"Equipo B: Varianza={np.var(equipo_B, ddof=1):.2f}, Desv={np.std(equipo_B, ddof=1):.2f}")
print("El Equipo A tiene mayor dispersión que el Equipo B.")

# ============================================================
# EJERCICIO 19: AGUA UTILIZADA
# ============================================================
print("\n--- Ejercicio 19: Agua utilizada ---")
agua = np.array([12, 15, 14, 10, 13, 11])
print(f"Media: {np.mean(agua):.2f}")
print(f"Varianza muestral: {np.var(agua, ddof=1):.2f}")
print(f"Desviación estándar muestral: {np.std(agua, ddof=1):.2f}")

# ============================================================
# EJERCICIO 20: FUNCIÓN y = 2x + 3
# ============================================================
print("\n--- Ejercicio 20: Función y = 2x + 3 ---")
x_vals = np.array([2, -1, 0, 1, 2, 3])
y_vals = 2*x_vals + 3
print("x:", x_vals)
print("y:", y_vals)
plt.figure()
plt.plot(x_vals, y_vals, marker='o', color='purple', label='y = 2x + 3')
plt.title('Ejercicio 20: y = 2x + 3')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
plt.savefig('ej20_funcion.png'); plt.close()
print("Gráfica guardada: ej20_funcion.png")
print(f"Pendiente: 2 (y aumenta 2 por cada unidad de x)")
print(f"Intersección con eje y: 3 (punto (0,3))")

print("\n" + "="*60)
print("FIN DE LA EJECUCIÓN")
print("="*60)