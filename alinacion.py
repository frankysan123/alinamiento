import streamlit as st
import math
import matplotlib.pyplot as plt

# --- Función principal ---
def distancia_perpendicular_invertida(A, B, P, tolerancia=0.01):
    xA, yA = A
    xB, yB = B
    xP, yP = P

    dx = xB - xA
    dy = yB - yA
    dx_p = xP - xA
    dy_p = yP - yA

    # Determinante (producto cruzado)
    det = dx * dy_p - dy * dx_p
    AB = math.sqrt(dx**2 + dy**2)

    # Distancia perpendicular con signo invertido
    d = -det / AB  

    # Proyección de P sobre AB
    t = (dx_p*dx + dy_p*dy) / (dx**2 + dy**2)
    proj_x = xA + t*dx
    proj_y = yA + t*dy

    # Vector de corrección
    corr_x = proj_x - xP
    corr_y = proj_y - yP

    alineado = abs(d) <= tolerancia

    return d, alineado, (proj_x, proj_y), (corr_x, corr_y)

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Calculadora de Alineación", page_icon="📐")

st.title("📐 Calculadora de Alineación (Topografía)")
st.write("Verifica si un punto **P** está alineado con la línea **AB** y observa la geometría en el gráfico.")

# Entradas de usuario
st.subheader("Coordenadas de los puntos")
xA = float(st.text_input("X A", value=1072.998, format="%.3f")
yA = float(st.text_input("Y A", value=971.948, format="%.3f")
xB = float(st.text_input("X B", value=963.595, format="%.3f")
yB = float(st.text_inputt("Y B", value=1012.893, format="%.3f")
xP = float(st.text_input("X PI", value=1040.749, format="%.3f")
yP = float(st.text_input("Y I", value=983.875, format="%.3f")
tol = float(st.text_input("Tolerancia (m)", value=0.01, format="%.3f")

# Botón de cálculo
if st.button("Calcular"):
    d, alineado, proyeccion, correccion = distancia_perpendicular_invertida(
        (xA,yA), (xB,yB), (xP,yP), tol
    )

    st.subheader("📊 Resultados")
    st.write(f"**Distancia perpendicular (con signo):** {d:.3f} m")
    st.write("**¿Está alineado?**", "✅ Sí" if alineado else "❌ No")
    st.write(f"**Proyección de P sobre AB:** ({proyeccion[0]:.3f}, {proyeccion[1]:.3f})")
    st.write(f"**Vector de corrección (mover P):** ΔX = {correccion[0]:.3f}, ΔY = {correccion[1]:.3f}")
    
    # Interpretación del signo
    if d > 0:
        st.info("➡️ P está a la **derecha** de la línea AB (mirando de A hacia B).")
    elif d < 0:
        st.info("⬅️ P está a la **izquierda** de la línea AB (mirando de A hacia B).")
    else:
        st.success("🎯 P está exactamente sobre la línea AB.")

    # --- Gráfico con Matplotlib ---
    fig, ax = plt.subplots()

    # Línea AB
    ax.plot([xA, xB], [yA, yB], 'b-', label="Línea AB")

    # Punto P
    ax.plot(xP, yP, 'ro', label="P (punto medido)")

    # Proyección
    ax.plot(proyeccion[0], proyeccion[1], 'go', label="Proyección de P")

    # Línea perpendicular desde P
    ax.plot([xP, proyeccion[0]], [yP, proyeccion[1]], 'r--', label="Perpendicular")

    # Estética
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Visualización geométrica")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    st.pyplot(fig)


