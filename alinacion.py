import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Alineación de PC con AB", 
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
    .division-box {
        background-color: #e8f4fd;
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
st.markdown('<div class="main-header">📐 Verificación de Alineación de Punto de Control con Línea AB + División de Segmentos</div>', unsafe_allow_html=True)
st.markdown("Introduce las coordenadas de dos puntos **A y B** y un punto **PC** (Punto de Control).")

# Sidebar for inputs
with st.sidebar:
    st.header("🔧 Parámetros de Entrada")
    
    st.subheader("Coordenadas del Punto A")
    xA = float(st.text_input("Coordenada X A", value="1072.998"))
    yA = float(st.text_input("Coordenada Y A", value="971.948"))
    
    st.subheader("Coordenadas del Punto B")
    xB = float(st.text_input("Coordenada X B", value="963.595"))
    yB = float(st.text_input("Coordenada Y B", value="1012.893"))
    
    st.subheader("Coordenadas del Punto PC")
    xPC = float(st.text_input("Coordenada X PC", value="1040.749"))
    yPC = float(st.text_input("Coordenada Y PC", value="983.875"))
    
    tol = st.slider("Tolerancia (m)", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=0.01, 
                    step=0.001, 
                    format="%.3f")
    
    # New: Segment division option
    st.subheader("🔢 División del Segmento AB")
    num_divisions = st.number_input("Número de divisiones", 
                                   min_value=1, 
                                   max_value=20, 
                                   value=5, 
                                   step=1,
                                   help="Divide el segmento AB en partes iguales")

# --- Functions ---
def distancia_perpendicular(A, B, PC):
    (xA, yA), (xB, yB), (xPC, yPC) = A, B, PC
    det = (xB - xA)*(yA - yPC) - (yB - yA)*(xA - xPC)
    AB = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    if AB == 0:
        return float('inf')  # Points A and B are the same
    d = -det / AB  # positivo = derecha, negativo = izquierda
    return d

def proyeccion(A, B, PC):
    A = np.array(A)
    B = np.array(B)
    PC = np.array(PC)
    AB = B - A
    AP = PC - A
    dot_product = np.dot(AP, AB)
    if np.dot(AB, AB) == 0:
        return A  # Points A and B are the same
    t = dot_product / np.dot(AB, AB)
    return A + t*AB

def calcular_distancia(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def dividir_segmento(A, B, num_partes):
    """
    Divide el segmento AB en num_partes iguales
    Retorna lista de puntos incluyendo A y B
    """
    A = np.array(A)
    B = np.array(B)
    puntos = []
    
    for i in range(num_partes + 1):
        t = i / num_partes
        punto = A + t * (B - A)
        puntos.append((float(punto[0]), float(punto[1])))
    
    return puntos

# --- Calculations ---
A = (xA, yA)
B = (xB, yB)
PC = (xPC, yPC)

# Validate that A and B are not the same point
if calcular_distancia(A, B) < 0.001:
    st.error("❌ Los puntos A y B son demasiado cercanos o iguales. Por favor, ingrese puntos distintos.")
    st.stop()

d_signed = distancia_perpendicular(A, B, PC)
d_abs = abs(d_signed)
proj = proyeccion(A, B, PC)
corr_vector = proj - np.array(PC)
alineado = d_abs <= tol
dist_perp = calcular_distancia(PC, proj)
dist_AB = calcular_distancia(A, B)

# Calculate division points
puntos_division = dividir_segmento(A, B, num_divisions)
longitud_entre_puntos = dist_AB / num_divisions

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
        st.success(f"✅ **PC está ALINEADO** con AB (dentro de la tolerancia de {tol} m)")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"⚠️ **PC NO está alineado** con AB (fuera de tolerancia de {tol} m)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Direction indicator
    if d_signed > 0:
        st.info("📍 **PC está a la DERECHA** de la línea AB")
    elif d_signed < 0:
        st.info("📍 **PC está a la IZQUIERDA** de la línea AB")
    else:
        st.info("🎯 **PC está exactamente sobre** la línea AB")
    
    # Projection details
    st.subheader("📐 Detalles de Proyección")
    st.write(f"**Coordenadas de proyección:** ({proj[0]:.3f}, {proj[1]:.3f})")
    st.write(f"**Vector de corrección:** ΔX = {corr_vector[0]:.3f} m, ΔY = {corr_vector[1]:.3f} m")
    
    # Division results
    st.subheader("📏 División del Segmento AB")
    st.markdown('<div class="division-box">', unsafe_allow_html=True)
    st.write(f"**Segmento AB dividido en {num_divisions} partes iguales**")
    st.write(f"**Longitud entre puntos:** {longitud_entre_puntos:.3f} m")
    st.write(f"**Total de puntos generados:** {len(puntos_division)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Division points table
    st.subheader("📋 Coordenadas de los Puntos de División")
    division_data = []
    for i, punto in enumerate(puntos_division):
        distancia_desde_A = calcular_distancia(A, punto)
        division_data.append({
            "Punto": f"P{i}",
            "X": f"{punto[0]:.3f}",
            "Y": f"{punto[1]:.3f}",
            "Distancia desde A": f"{distancia_desde_A:.3f} m"
        })
    
    # Show first few points with expander for all points
    st.table(division_data[:6])  # Show first 6 points
    
    if len(division_data) > 6:
        with st.expander("Ver todos los puntos"):
            for i in range(6, len(division_data)):
                punto = division_data[i]
                st.write(f"{punto['Punto']}: X={punto['X']}, Y={punto['Y']}, Distancia A={punto['Distancia desde A']}")

with col2:
    st.subheader("📈 Visualización Gráfica")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Line AB
    ax.plot([xA, xB], [yA, yB], 'b-', linewidth=3, label="Línea AB", alpha=0.7)
    
    # Perpendicular line
    ax.plot([xPC, proj[0]], [yPC, proj[1]], 'r--', linewidth=2, label="Distancia perpendicular", alpha=0.7)
    
    # Division points
    division_x = [p[0] for p in puntos_division]
    division_y = [p[1] for p in puntos_division]
    ax.scatter(division_x, division_y, c='orange', s=50, alpha=0.7, label=f"Puntos de división ({num_divisions} partes)")
    
    # Label division points
    for i, (x, y) in enumerate(puntos_division):
        if i == 0:  # Point A
            ax.text(x, y, '  A', verticalalignment='center', fontweight='bold', fontsize=10)
        elif i == len(puntos_division) - 1:  # Point B
            ax.text(x, y, '  B', verticalalignment='center', fontweight='bold', fontsize=10)
        else:
            ax.text(x, y, f'  P{i}', verticalalignment='center', fontsize=8, alpha=0.8)
    
    # Points with enhanced styling
    # Point A (already labeled above)
    ax.plot(xA, yA, 'bo', markersize=8)
    
    # Point B (already labeled above)
    ax.plot(xB, yB, 'bo', markersize=8)
    
    # Distance annotation with arrow
    mid_x = (xPC + proj[0]) / 2
    mid_y = (yPC + proj[1]) / 2
    
    # Add perpendicular distance annotation
    ax.annotate('', xy=(proj[0], proj[1]), xytext=(xPC, yPC),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=1.5))
    
    # Point PC
    ax.plot(xPC, yPC, 'ro', markersize=12, markerfacecolor='red', label="Punto de Control (PC)")
    
    
    # Projection point
    ax.plot(proj[0], proj[1], 'gs', markersize=10, label="Proyección")
   

    # Distance label
    offset_x = 7
    offset_y = 7
    ax.text(mid_x + offset_x, mid_y + offset_y, f'd = {dist_perp:.3f} m',
            backgroundcolor='white', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Adjust plot limits with margin
    margin = max(dist_perp, dist_AB * 0.1) + 2
    min_x = min(xA, xB, xPC, proj[0]) - margin
    max_x = max(xA, xB, xPC, proj[0]) + margin
    min_y = min(yA, yB, yPC, proj[1]) - margin
    max_y = max(yA, yB, yPC, proj[1]) + margin
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Plot aesthetics
    ax.set_xlabel("Coordenada X (m)")
    ax.set_ylabel("Coordenada Y (m)")
    ax.set_title(f"Visualización de Alineación PC-AB + División en {num_divisions} Partes", 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    st.pyplot(fig)

# Additional information
st.subheader("📋 Información Adicional")
col3, col4 = st.columns(2)

with col3:
    st.write("**Interpretación de resultados:**")
    st.write("- **Distancia positiva**: PC a la derecha de AB")
    st.write("- **Distancia negativa**: PC a la izquierda de AB")
    st.write("- **Distancia cero**: PC sobre la línea AB")
    st.write("**División del segmento:**")
    st.write(f"- P0 = Punto A")
    st.write(f"- P{num_divisions} = Punto B")
    st.write(f"- Cada segmento mide {longitud_entre_puntos:.3f} m")

with col4:
    st.write("**Recomendaciones:**")
    st.write("- Ajuste la tolerancia según la precisión requerida")
    st.write("- Verifique que los puntos A y B sean distintos")
    st.write("- Use el vector de corrección para ajustar la posición")
    st.write("- Los puntos de división son útiles para:")
    st.write("  • Estacas intermedias en topografía")
    st.write("  • Puntos de referencia en construcción")
    st.write("  • Muestreo equidistante a lo largo de AB")

# Footer
st.markdown("---")
st.markdown("*Herramienta desarrollada para verificación de alineación topográfica y división de segmentos*")

