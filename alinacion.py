import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Alineación de PC con AB", 
    layout="wide",
    page_icon="📐",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.5rem !important;
        }
        .stColumn {
            min-width: 100% !important;
        }
    }
    
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
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
    .stDownloadButton button {
        width: 100%;
    }
    
    /* Optimización para móvil */
    @media (max-width: 640px) {
        .stPlotlyChart {
            height: 500px !important;
        }
    }
    
    /* Mejor contraste en botones */
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton>button:hover {
        background-color: #1557a0;
        border-color: #1557a0;
    }
    
    /* Mejora en inputs */
    .stTextInput>div>div>input {
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Sidebar más compacto */
    .css-1d391kg {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history
if 'calculation_history' not in st.session_state:
    st.session_state.calculation_history = []

# Header
st.markdown('<div class="main-header">📐 Verificación de Alineación de Punto de Control con Línea AB + División de Segmentos</div>', unsafe_allow_html=True)
st.markdown("Introduce las coordenadas de dos puntos **A y B** y un punto **PC** (Punto de Control).")

# --- Functions with caching ---
@st.cache_data
def distancia_perpendicular(A, B, PC):
    """Calcula la distancia perpendicular con signo (positivo=derecha, negativo=izquierda)"""
    (xA, yA), (xB, yB), (xPC, yPC) = A, B, PC
    det = (xB - xA)*(yA - yPC) - (yB - yA)*(xA - xPC)
    AB = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    if AB == 0:
        return float('inf')
    d = -det / AB
    return d

@st.cache_data
def proyeccion(A, B, PC):
    """Proyecta el punto PC sobre la línea AB"""
    A = np.array(A)
    B = np.array(B)
    PC = np.array(PC)
    AB = B - A
    AP = PC - A
    dot_product = np.dot(AP, AB)
    if np.dot(AB, AB) == 0:
        return A
    t = dot_product / np.dot(AB, AB)
    return A + t*AB

@st.cache_data
def calcular_distancia(p1, p2):
    """Calcula distancia euclidiana entre dos puntos"""
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

@st.cache_data
def dividir_segmento(A, B, num_partes):
    """Divide el segmento AB en num_partes iguales"""
    A = np.array(A)
    B = np.array(B)
    puntos = []
    
    for i in range(num_partes + 1):
        t = i / num_partes
        punto = A + t * (B - A)
        puntos.append((float(punto[0]), float(punto[1])))
    
    return puntos

def validar_coordenada(valor, nombre):
    """Valida que el valor sea numérico"""
    try:
        return float(valor)
    except (ValueError, TypeError):
        st.error(f"❌ Error: {nombre} debe ser un valor numérico")
        st.stop()

def crear_dataframe_division(puntos_division, A):
    """Crea DataFrame con información de puntos de división"""
    data = []
    for i, punto in enumerate(puntos_division):
        distancia_desde_A = calcular_distancia(A, punto)
        data.append({
            "Punto": f"P{i}",
            "X": round(punto[0], 3),
            "Y": round(punto[1], 3),
            "Distancia desde A (m)": round(distancia_desde_A, 3)
        })
    return pd.DataFrame(data)

def exportar_excel(df_division, resultados):
    """Exporta datos a Excel con múltiples hojas"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Hoja de puntos de división
        df_division.to_excel(writer, sheet_name='Puntos División', index=False)
        
        # Hoja de resultados
        df_resultados = pd.DataFrame([resultados])
        df_resultados.to_excel(writer, sheet_name='Resultados', index=False)
    
    output.seek(0)
    return output

def crear_grafico_plotly(A, B, PC, proj, puntos_division, d_signed, dist_perp, num_divisions):
    """Crea gráfico interactivo con Plotly optimizado para móvil"""
    fig = go.Figure()
    
    # Línea AB
    fig.add_trace(go.Scatter(
        x=[A[0], B[0]], 
        y=[A[1], B[1]],
        mode='lines',
        name='Línea AB',
        line=dict(color='blue', width=4),
        hovertemplate='<b>Línea AB</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))
    
    # Línea perpendicular
    fig.add_trace(go.Scatter(
        x=[PC[0], proj[0]], 
        y=[PC[1], proj[1]],
        mode='lines',
        name='Dist. perpendicular',
        line=dict(color='red', width=3, dash='dash'),
        hovertemplate=f'<b>Distancia</b><br>{dist_perp:.3f} m<extra></extra>'
    ))
    
    # Puntos de división
    division_x = [p[0] for p in puntos_division]
    division_y = [p[1] for p in puntos_division]
    division_labels = [f'P{i}' for i in range(len(puntos_division))]
    
    fig.add_trace(go.Scatter(
        x=division_x,
        y=division_y,
        mode='markers+text',
        name=f'División ({num_divisions})',
        marker=dict(color='orange', size=10),
        text=division_labels,
        textposition='top center',
        textfont=dict(size=10),
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))
    
    # Punto A
    fig.add_trace(go.Scatter(
        x=[A[0]], y=[A[1]],
        mode='markers+text',
        name='Punto A',
        marker=dict(color='blue', size=16, symbol='circle'),
        text=['A'],
        textposition='bottom center',
        textfont=dict(size=14, color='blue'),
        hovertemplate='<b>Punto A</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))
    
    # Punto B
    fig.add_trace(go.Scatter(
        x=[B[0]], y=[B[1]],
        mode='markers+text',
        name='Punto B',
        marker=dict(color='blue', size=16, symbol='circle'),
        text=['B'],
        textposition='bottom center',
        textfont=dict(size=14, color='blue'),
        hovertemplate='<b>Punto B</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))
    
    # Punto PC
    fig.add_trace(go.Scatter(
        x=[PC[0]], y=[PC[1]],
        mode='markers+text',
        name='Punto Control',
        marker=dict(color='red', size=18, symbol='diamond'),
        text=['PC'],
        textposition='top center',
        textfont=dict(size=14, color='red'),
        hovertemplate='<b>PC</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))
    
    # Proyección
    fig.add_trace(go.Scatter(
        x=[proj[0]], y=[proj[1]],
        mode='markers+text',
        name='Proyección',
        marker=dict(color='green', size=14, symbol='square'),
        text=['Proj'],
        textposition='bottom center',
        textfont=dict(size=12, color='green'),
        hovertemplate='<b>Proyección</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
    ))
    
    # Layout optimizado para móvil y web
    fig.update_layout(
        title={
            'text': f'Alineación PC-AB (División: {num_divisions})',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        hovermode='closest',
        height=700,
        margin=dict(l=20, r=20, t=60, b=100),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        dragmode='pan',
        # Optimización para móvil
        autosize=True,
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)')
    
    # Configuración para mejor experiencia táctil en móvil
    fig.update_layout(
        modebar=dict(
            orientation='v',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )
    
    return fig

# Sidebar for inputs
with st.sidebar:
    st.header("🔧 Parámetros de Entrada")
    
    # Modo de entrada
    modo_entrada = st.radio(
        "Modo de entrada de datos",
        ["Manual", "Cargar desde archivo CSV"],
        help="Selecciona cómo ingresar las coordenadas"
    )
    
    if modo_entrada == "Cargar desde archivo CSV":
        st.info("📁 El archivo CSV debe tener columnas: Punto, X, Y")
        uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if len(df) >= 3:
                    xA, yA = df.iloc[0]['X'], df.iloc[0]['Y']
                    xB, yB = df.iloc[1]['X'], df.iloc[1]['Y']
                    xPC, yPC = df.iloc[2]['X'], df.iloc[2]['Y']
                    st.success("✅ Archivo cargado correctamente")
                else:
                    st.error("El archivo debe tener al menos 3 puntos")
                    st.stop()
            except Exception as e:
                st.error(f"Error al cargar archivo: {e}")
                st.stop()
        else:
            st.warning("Por favor, carga un archivo CSV")
            st.stop()
    else:
        st.subheader("Coordenadas del Punto A")
        xA = validar_coordenada(st.text_input("Coordenada X A", value="1072.998"), "Coordenada X A")
        yA = validar_coordenada(st.text_input("Coordenada Y A", value="971.948"), "Coordenada Y A")
        
        st.subheader("Coordenadas del Punto B")
        xB = validar_coordenada(st.text_input("Coordenada X B", value="963.595"), "Coordenada X B")
        yB = validar_coordenada(st.text_input("Coordenada Y B", value="1012.893"), "Coordenada Y B")
        
        st.subheader("Coordenadas del Punto PC")
        xPC = validar_coordenada(st.text_input("Coordenada X PC", value="1040.749"), "Coordenada X PC")
        yPC = validar_coordenada(st.text_input("Coordenada Y PC", value="983.875"), "Coordenada Y PC")
    
    tol = st.slider("Tolerancia (m)", 
                    min_value=0.001, 
                    max_value=1.0, 
                    value=0.01, 
                    step=0.001, 
                    format="%.3f")
    
    st.subheader("🔢 División del Segmento AB")
    num_divisions = st.number_input("Número de divisiones", 
                                   min_value=1, 
                                   max_value=50, 
                                   value=5, 
                                   step=1,
                                   help="Divide el segmento AB en partes iguales")
    
    st.markdown("---")
    formato_export = st.selectbox(
        "💾 Formato de exportación",
        ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
    )

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

# Create DataFrame for division points
df_division = crear_dataframe_division(puntos_division, A)

# Prepare results dictionary
resultados = {
    'Fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'A_X': xA, 'A_Y': yA,
    'B_X': xB, 'B_Y': yB,
    'PC_X': xPC, 'PC_Y': yPC,
    'Distancia_Perpendicular': round(d_signed, 3),
    'Distancia_Absoluta': round(d_abs, 3),
    'Distancia_AB': round(dist_AB, 3),
    'Tolerancia': tol,
    'Alineado': 'Sí' if alineado else 'No',
    'Posicion': 'Derecha' if d_signed > 0 else ('Izquierda' if d_signed < 0 else 'Sobre línea'),
    'Proyeccion_X': round(proj[0], 3),
    'Proyeccion_Y': round(proj[1], 3),
    'Num_Divisiones': num_divisions,
    'Longitud_Entre_Puntos': round(longitud_entre_puntos, 3)
}

# Add to history
st.session_state.calculation_history.append(resultados)

# --- Results Display ---
col1, col2 = st.columns([1.4, 1])

with col1:
    st.subheader("📈 Visualización Gráfica Interactiva")
    
    # Create interactive Plotly chart
    fig = crear_grafico_plotly(A, B, PC, proj, puntos_division, d_signed, dist_perp, num_divisions)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📊 Resultados de Alineación")
    
    # Distance results
    with st.container():
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Distancia perpendicular", f"{d_abs:.3f} m")
        with col_m2:
            st.metric("Distancia AB", f"{dist_AB:.3f} m")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Position indicator
    if alineado:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success(f"✅ **PC está ALINEADO** con AB (tolerancia: {tol} m)")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning(f"⚠️ **PC NO está alineado** con AB (tolerancia: {tol} m)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Direction indicator
    if d_signed > 0:
        st.info(f"📍 **PC está a la DERECHA** de AB ({d_signed:.3f} m)")
    elif d_signed < 0:
        st.info(f"📍 **PC está a la IZQUIERDA** de AB ({abs(d_signed):.3f} m)")
    else:
        st.info("🎯 **PC está exactamente sobre** la línea AB")
    
    # Projection details
    with st.expander("📐 Ver Detalles de Proyección"):
        st.write(f"**Coordenadas de proyección:** ({proj[0]:.3f}, {proj[1]:.3f})")
        st.write(f"**Vector de corrección:** ΔX = {corr_vector[0]:.3f} m, ΔY = {corr_vector[1]:.3f} m")
        st.write(f"**Magnitud corrección:** {np.linalg.norm(corr_vector):.3f} m")
    
    # Division results
    st.subheader("📏 División del Segmento AB")
    st.markdown('<div class="division-box">', unsafe_allow_html=True)
    st.write(f"**Segmento dividido en:** {num_divisions} partes iguales")
    st.write(f"**Longitud entre puntos:** {longitud_entre_puntos:.3f} m")
    st.write(f"**Total de puntos:** {len(puntos_division)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Division points table
    st.subheader("📋 Tabla de Puntos de División")
    st.dataframe(df_division, use_container_width=True, height=400)

# Export section - MOVED AFTER CALCULATIONS
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Descargar Resultados")

if formato_export == "Excel (.xlsx)":
    excel_data = exportar_excel(df_division, resultados)
    st.sidebar.download_button(
        label="📥 Descargar Excel",
        data=excel_data,
        file_name=f"topo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
elif formato_export == "CSV (.csv)":
    csv_data = df_division.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Descargar CSV",
        data=csv_data,
        file_name=f"topo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:  # JSON
    json_data = df_division.to_json(orient='records', indent=2)
    st.sidebar.download_button(
        label="📥 Descargar JSON",
        data=json_data,
        file_name=f"topo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )

# Additional information
st.markdown("---")
st.subheader("📋 Información Adicional")
col3, col4, col5 = st.columns(3)

with col3:
    st.write("**Interpretación:**")
    st.write("- ➕ Distancia positiva: PC a la derecha")
    st.write("- ➖ Distancia negativa: PC a la izquierda")
    st.write("- 0️⃣ Distancia cero: PC sobre AB")

with col4:
    st.write("**División:**")
    st.write(f"- P0 = Punto A")
    st.write(f"- P{num_divisions} = Punto B")
    st.write(f"- Cada segmento: {longitud_entre_puntos:.3f} m")

with col5:
    st.write("**Recomendaciones:**")
    st.write("- Ajuste tolerancia según precisión")
    st.write("- Use vector de corrección")
    st.write("- Exporte datos para reportes")

# Calculation history
with st.expander("📜 Ver Historial de Cálculos (Sesión Actual)"):
    if len(st.session_state.calculation_history) > 0:
        df_history = pd.DataFrame(st.session_state.calculation_history)
        st.dataframe(df_history, use_container_width=True)
        
        if st.button("🗑️ Limpiar Historial"):
            st.session_state.calculation_history = []
            st.rerun()
    else:
        st.info("No hay cálculos en el historial aún")

# Footer
st.markdown("---")
st.markdown("*Herramienta mejorada para verificación de alineación topográfica y división de segmentos*")
st.markdown("**Versión 2.0** - Con exportación de datos, gráficos interactivos y caché optimizado")
