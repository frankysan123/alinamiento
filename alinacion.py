import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Alineación de PT con AB", 
    layout="wide",
    page_icon="📐"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    /* Hide number input arrows */
    input[type=number]::-webkit-outer-spin-button,
    input[type=number]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    input[type=number] {
        -moz-appearance: textfield;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📐 Verificación de Alineación de Punto con Línea AB</div>', unsafe_allow_html=True)
st.markdown("Introduce las coordenadas de dos puntos **A y B** y un punto **PT** (Punto de Trabajo).")

# Sidebar for inputs
with st.sidebar:
    st.header("🔧 Parámetros de Entrada")
    
    st.subheader("Coordenadas del Punto A")
    xA = float(st.text_input("Coordenada X A", value="1072.998"))
    yA = float(st.text_input("Coordenada Y A", value="971.948"))
    
    st.subheader("Coordenadas del Punto B")
    xB = float(st.text_input("Coordenada X B", value="963.595"))
    yB = float(st.text_input("Coordenada Y B", value="1012.893"))
    
    st.subheader("Coordenadas del Punto PT")
    xPT = float(st.text_input("Coordenada X PT", value="1040.749"))
    yPT = float(st.text_input("Coordenada Y PT", value="983.875"))
    
    tol = st.slider("Tolerancia (m)", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=0.01, 
                    step=0.001, 
                    format="%.3f")

# --- Functions ---
def distancia_perpendicular(A, B, PT):
    (xA, yA), (xB, yB), (xPT, yPT) = A, B, PT
    det = (xB - xA)*(yA - yPT) - (yB - yA)*(xA - xPT)
    AB = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    if AB == 0:
        return float('inf')  # Points A and B are the same
    d = -det / AB  # positivo = derecha, negativo = izquierda
    return d

def proyeccion(A, B, PT):
    A = np.array(A)
    B = np.array(B)
    PT = np.array(PT)
    AB = B - A
    AP = PT - A
    dot_product = np.dot(AP, AB)
    if np.dot(AB, AB) == 0:
        return A  # Points A and B are the same
    t = dot_product / np.dot(AB, AB)
    return A + t*AB

def calcular_distancia(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# --- Calculations ---
A = (xA, yA)
B = (xB, yB)
PT = (xPT, yPT)

# Validate that A and B are not the same point
if calcular_distancia(A, B) < 0.001:
    st.error("❌ Los puntos A y B son demasiado cercanos o iguales. Por favor, ingrese puntos distintos.")
    st.stop()

d_signed = distancia_perpendicular(A, B, PT)
d_abs = abs(d_signed)
proj = proyeccion(A, B, PT)
corr_vector = proj - np.array(PT)
alineado = d_abs <= tol
dist_perp = calcular_distancia(PT, proj)
dist_AB = calcular_distancia(A, B)

# --- Results Display ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Resultados de Alineación")
    
    # Distance results
    with st.container():
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.metric("Distancia perpendicular absoluta", f"{d_abs:.3f} m", delta=f"{d_signed:.3f} m")
        st.metric("Distancia del segmento AB", f"{dist_AB:.3f} m")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Position indicator
    if alineado:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(f"✅ **PT está ALINEADO** con AB (dentro de la tolerancia de {tol} m)")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"⚠️ **PT NO está alineado** con AB (fuera de tolerancia de {tol} m)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Direction indicator
    if d_signed > 0:
        st.info("📍 **PT está a la DERECHA** de la línea AB")
    elif d_signed < 0:
        st.info("📍 **PT está a la IZQUIERDA** de la línea AB")
    else:
        st.info("🎯 **PT está exactamente sobre** la línea AB")
    
    # Projection details
    st.subheader("📐 Detalles de Proyección")
    st.write(f"**Coordenadas de proyección:** ({proj[0]:.3f}, {proj[1]:.3f})")
    st.write(f"**Vector de corrección:** ΔX = {corr_vector[0]:.3f} m, ΔY = {corr_vector[1]:.3f} m")

with col2:
    st.subheader("📈 Visualización Gráfica")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Line AB
    ax.plot([xA, xB], [yA, yB], 'b-', linewidth=2, label="Línea AB", alpha=0.7)
    
    # Perpendicular line
    ax.plot([xPT, proj[0]], [yPT, proj[1]], 'r--', linewidth=2, label="Distancia perpendicular", alpha=0.7)
    
    # Points with enhanced styling
    # Point A
    ax.plot(xA, yA, 'bo', markersize=10, label="Punto A")
    ax.text(xA, yA, '  A', verticalalignment='center', fontweight='bold')
    
    # Point B
    ax.plot(xB, yB, 'bo', markersize=10, label="Punto B")
    ax.text(xB, yB, '  B', verticalalignment='center', fontweight='bold')
    
    # Point PT
    ax.plot(xPT, yPT, 'ro', markersize=12, markerfacecolor='red', label="Punto PT")
    ax.text(xPT, yPT, '  PT', verticalalignment='center', fontweight='bold', color='red')
    
    # Projection point
    ax.plot(proj[0], proj[1], 'gs', markersize=10, label="Proyección")
    ax.text(proj[0], proj[1], '  Proyección', verticalalignment='center', fontweight='bold', color='green')
    
    # Distance annotation with arrow
    mid_x = (xPT + proj[0]) / 2
    mid_y = (yPT + proj[1]) / 2
    
  # Add perpendicular distance annotation
ax.annotate('', xy=(proj[0], proj[1]), xytext=(xPT, yPT),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))

# Desplazar un poco la etiqueta (ejemplo: +0.5 en Y)
offset_x = 0.5
offset_y = 0.5

ax.text(mid_x + offset_x, mid_y + offset_y, f'd = {dist_perp:.3f} m',
        backgroundcolor='white', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    
    # Adjust plot limits with margin
    margin = max(dist_perp, dist_AB * 0.1) + 2
    min_x = min(xA, xB, xPT, proj[0]) - margin
    max_x = max(xA, xB, xPT, proj[0]) + margin
    min_y = min(yA, yB, yPT, proj[1]) - margin
    max_y = max(yA, yB, yPT, proj[1]) + margin
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Plot aesthetics
    ax.set_xlabel("Coordenada X (m)")
    ax.set_ylabel("Coordenada Y (m)")
    ax.set_title("Visualización de Alineación PT-AB", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    st.pyplot(fig)

# Additional information
st.subheader("📋 Información Adicional")
col3, col4 = st.columns(2)

with col3:
    st.write("**Interpretación de resultados:**")
    st.write("- **Distancia positiva**: PT a la derecha de AB")
    st.write("- **Distancia negativa**: PT a la izquierda de AB")
    st.write("- **Distancia cero**: PT sobre la línea AB")

with col4:
    st.write("**Recomendaciones:**")
    st.write("- Ajuste la tolerancia según la precisión requerida")
    st.write("- Verifique que los puntos A y B sean distintos")
    st.write("- Use el vector de corrección para ajustar la posición")

# Footer
st.markdown("---")
st.markdown("*Herramienta desarrollada para verificación de alineación topográfica*")





