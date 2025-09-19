import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Alineación de PT con AB", layout="centered")

st.title("📐 Verificación de alineación de un punto con la línea AB")
st.markdown("Introduce las coordenadas de dos puntos **A y B** y un punto **PT** (Punto de Trabajo).")

# --- Entradas de usuario
xA = float(st.text_input("X de A", value="1072.998"))
yA = float(st.text_input("Y de A", value="971.948"))
xB = float(st.text_input("X de B", value="963.595"))
yB = float(st.text_input("Y de B", value="1012.893"))
xPT = float(st.text_input("X de PT", value="1040.749"))
yPT = float(st.text_input("Y de PT", value="983.875"))
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
st.subheader("📈 Visualización")
fig, ax = plt.subplots(figsize=(7,7))

# Línea AB
ax.plot([xA, xB], [yA, yB], 'b-', linewidth=2, label="Línea AB")
# Línea perpendicular
ax.plot([xPT, proj[0]], [yPT, proj[1]], 'r--', linewidth=2, label="Perpendicular")
# Punto PT
ax.plot(xPT, yPT, 'ro', markersize=10, label="PT")
# Proyección
ax.plot(proj[0], proj[1], 'go', markersize=10, label="Proyección de PT")
# Etiquetas
ax.text(xPT, yPT, " PT", color='red', fontsize=12, fontweight='bold', ha='right', va='bottom')
ax.text(proj[0], proj[1], " Proy", color='green', fontsize=12, ha='left', va='bottom')

# Distancia perpendicular en el gráfico
mid_x = (xPT + proj[0]) / 2
mid_y = (yPT + proj[1]) / 2
ax.text(mid_x, mid_y, f"{dist_perp:.3f} m", color='purple', fontsize=10, fontweight='bold')

# Ajustes estéticos
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Alineación de PT respecto a AB")
ax.grid(True)
ax.axis("equal")
ax.set_xlim(min(xA, xB, xPT, proj[0])-5, max(xA, xB, xPT, proj[0])+5)
ax.set_ylim(min(yA, yB, yPT, proj[1])-5, max(yA, yB, yPT, proj[1])+5)
ax.legend()

st.pyplot(fig)
