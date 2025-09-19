import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alineación de PT con AB", layout="centered")

st.title("📐 Verificación de alineación de un punto con la línea AB")
st.markdown("Introduce las coordenadas de dos puntos **A y B** y un punto **PT** (Punto de Trabajo).")

# --- Entradas de usuario
xA = float(st.text_input("CORD X A", value="1072.998"))
yA = float(st.text_input("CORD Y A", value="971.948"))
xB = float(st.text_input("CORD X B", value="963.595"))
yB = float(st.text_input("CORD Y B", value="1012.893"))
xPT = float(st.text_input("CORD X PI", value="1040.749"))
yPT = float(st.text_input("CORD Y PI", value="983.875"))
tol = float(st.text_input("Tolerancia (m)", value="0.01"))

# --- Funciones
def distancia_perpendicular(A, B, PT):
    (xA, yA), (xB, yB), (xPT, yPT) = A, B, PT
    det = (xB - xA)*(yA - yPT) - (yB - yA)*(xA - xPT)
    AB = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    d = -det / AB  # positivo = derecha, negativo = izquierda
    return d

def proyeccion(A, B, PT):
    A = np.array(A)
    B = np.array(B)
    PT = np.array(PT)
    AB = B - A
    AP = PT - A
    t = np.dot(AP, AB) / np.dot(AB, AB)
    return A + t*AB

# --- Cálculos
A = (xA, yA)
B = (xB, yB)
PT = (xPT, yPT)

d_signed = distancia_perpendicular(A, B, PT)
d_abs = abs(d_signed)
proj = proyeccion(A, B, PT)
corr_vector = proj - np.array(PT)
alineado = d_abs <= tol
dist_perp = np.sqrt((proj[0]-xPT)**2 + (proj[1]-yPT)**2)

# --- Resultados
st.subheader("📊 Resultados")
st.write(f"Distancia perpendicular (con signo) = **{d_signed:.3f} m**")
if d_signed > 0:
    st.success("➡️ PT está a la **derecha** de la línea AB")
elif d_signed < 0:
    st.warning("⬅️ PT está a la **izquierda** de la línea AB")
else:
    st.info("🎯 PT está exactamente sobre la línea AB")

st.write(f"Coordenadas de la proyección sobre AB: **({proj[0]:.3f}, {proj[1]:.3f})**")
st.write(f"Vector de corrección: ΔX = {corr_vector[0]:.3f}, ΔY = {corr_vector[1]:.3f}")

# --- Gráfico Mejorado
st.subheader("📈 Visualización Mejorada")
fig, ax = plt.subplots(figsize=(8,8))  # tamaño más grande para zoom

# Línea AB
ax.plot([xA, xB], [yA, yB], 'b-', linewidth=2, label="Línea AB")
# Línea perpendicular
ax.plot([xPT, proj[0]], [yPT, proj[1]], 'r--', linewidth=2, label="Perpendicular")

# Punto PT: círculo con cruz
ax.plot(xPT, yPT, 'ro', markersize=12, markerfacecolor='none', label="PT")
ax.plot([xPT-0.5, xPT+0.5], [yPT, yPT], 'r', linewidth=2)  # cruz horizontal
ax.plot([xPT, xPT], [yPT-0.5, yPT+0.5], 'r', linewidth=2)  # cruz vertical

# Punto Proyección: círculo con cruz
ax.plot(proj[0], proj[1], 'go', markersize=12, markerfacecolor='none', label="Proyección de PT")
ax.plot([proj[0]-0.5, proj[0]+0.5], [proj[1], proj[1]], 'g', linewidth=2)
ax.plot([proj[0], proj[0]], [proj[1]-0.5, proj[1]+0.5], 'g', linewidth=2)

# Desplazamiento de etiquetas
offset = 1  # alejar etiquetas de los puntos
ax.text(xPT + offset, yPT + offset, "PT", color='red', fontsize=8, fontweight='bold')
ax.text(proj[0] + offset, proj[1] + offset, "Proy", color='green', fontsize=8, fontweight='bold')

# Distancia perpendicular
mid_x = (xPT + proj[0]) / 2
mid_y = (yPT + proj[1]) / 2
ax.text(mid_x, mid_y + offset, f"{dist_perp:.3f} m", color='purple', fontsize=8, fontweight='bold')

# Ajustes de zoom
margin = 1.5  # menos margen = más zoom
min_x = min(xA, xB, xPT, proj[0]) - margin
max_x = max(xA, xB, xPT, proj[0]) + margin
min_y = min(yA, yB, yPT, proj[1]) - margin
max_y = max(yA, yB, yPT, proj[1]) + margin
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

# Ejes y estética
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Alineación de PT respecto a AB", fontsize=12)
ax.grid(True)
ax.axis("equal")
ax.legend(fontsize=9)

st.pyplot(fig)




