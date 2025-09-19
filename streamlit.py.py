import streamlit as st
import math

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

    # Vector de corrección (hacia la línea)
    corr_x = proj_x - xP
    corr_y = proj_y - yP

    alineado = abs(d) <= tolerancia

    return d, alineado, (proj_x, proj_y), (corr_x, corr_y)

# --- Interfaz Streamlit ---
st.set_page_config(page_title="Calculadora de Alineación", page_icon="📐")

st.title("📐 Calculadora de Alineación (Topografía)")
st.write("Verifica si un punto **P** está alineado con la línea **AB**.")

# Entradas de usuario
st.subheader("Coordenadas de los puntos")
xA = st.number_input("X de A", value=1072.998, format="%.3f")
yA = st.number_input("Y de A", value=971.948, format="%.3f")
xB = st.number_input("X de B", value=963.595, format="%.3f")
yB = st.number_input("Y de B", value=1012.893, format="%.3f")
xP = st.number_input("X de P", value=1040.749, format="%.3f")
yP = st.number_input("Y de P", value=983.875, format="%.3f")
tol = st.number_input("Tolerancia (m)", value=0.01, format="%.3f")

# Botón para calcular
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