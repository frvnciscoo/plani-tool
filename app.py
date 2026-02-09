import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model

# --- 1. CONFIGURACIÓN Y DATOS ---
st.set_page_config(page_title="Opti-Plan Consolidado", layout="wide")

# Tasas de rendimiento base (Generic)
BASE_RATES = {
    'MAV': {'turno': 20, 'admin': 26},
    'MAS': {'turno': 15, 'admin': 20},
    'Celulosa': {'turno': 35, 'admin': 46},
    'Papel': {'turno': 10, 'admin': 13}
}

# Lista completa de productos específicos (Cliente_Producto)
# Esto permite que el modelo optimice cada uno por separado
SPECIFIC_PRODUCTS = [
    'Celulosa_CMPC', 'Celulosa_ARAUCO',
    'MAV_CMPC', 'MAV_ARAUCO',
    'MAS_CMPC', 'MAS_ARAUCO',
    'Papel_CMPC', 'Papel_ARAUCO'
]

DAYS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
SHIFTS_ORDER = ['Admin', 'T1', 'T2', 'T3']

# Función helper para obtener la tasa correcta dado un producto específico
def get_rate(specific_product, type_rate):
    # Extrae "Celulosa" de "Celulosa_CMPC"
    base_category = specific_product.split('_')[0] 
    return BASE_RATES[base_category][type_rate]

# --- 2. INTERFAZ DE USUARIO (STREAMLIT) ---
st.title("PLANIFICACIÓN SEMANAL DE CUADRILLAS")
st.markdown("### Vista Detallada por Cliente")

with st.sidebar:
    st.header("📦 Demanda Semanal")
    st.write("Ingrese la cantidad total a consolidar:")
    
    # Diccionario para guardar la demanda de cada producto específico
    demands = {}

    st.subheader("Prioritario")
    demands['Celulosa_CMPC'] = st.number_input("CELULOSA CMPC", min_value=0, value=200)
    demands['Celulosa_ARAUCO'] = st.number_input("CELULOSA ARAUCO", min_value=0, value=218)

    st.subheader("Otros Productos")
    col1, col2 = st.columns(2)
    with col1:
        demands['MAV_CMPC'] = st.number_input("MAV CMPC", min_value=0, value=20)
        demands['MAS_CMPC'] = st.number_input("MAS CMPC", min_value=0, value=15)
        demands['Papel_CMPC'] = st.number_input("PAPEL CMPC", min_value=0, value=10)
    with col2:
        demands['MAV_ARAUCO'] = st.number_input("MAV ARAUCO", min_value=0, value=24)
        demands['MAS_ARAUCO'] = st.number_input("MAS ARAUCO", min_value=0, value=6)
        demands['Papel_ARAUCO'] = st.number_input("PAPEL ARAUCO", min_value=0, value=0)

    run_optimization = st.button("🔄 Generar Planificación", type="primary")

# --- 3. LÓGICA DE OPTIMIZACIÓN (OR-TOOLS) ---
def solve_schedule(demand_dict):
    model = cp_model.CpModel()

    # Filtramos productos con demanda > 0 para no gastar recursos computacionales en vacíos
    active_products = [p for p in SPECIFIC_PRODUCTS if demand_dict[p] > 0]

    # Variables
    x = {} # Cuadrillas turno
    for d in range(len(DAYS)):
        for s in range(3): # T1, T2, T3
            for p in active_products:
                x[d, s, p] = model.NewIntVar(0, 3, f'x_{d}_{s}_{p}')
    
    y = {} # Cuadrilla Admin
    for d in range(len(DAYS)):
        for p in active_products:
            y[d, p] = model.NewBoolVar(f'y_{d}_{p}')
            
    three_crews_used = {} # Penalización
    for d in range(len(DAYS)):
        for s in range(3):
             three_crews_used[d, s] = model.NewBoolVar(f'3crews_{d}_{s}')

    # Restricciones
    for d in range(len(DAYS)):
        # Max 1 Admin por día (entre todos los clientes y productos)
        model.Add(sum(y[d, p] for p in active_products) <= 1)
        
        for s in range(3):
            # Max 3 cuadrillas por turno (SUMA de CMPC + ARAUCO + ETC)
            total_crews_in_shift = sum(x[d, s, p] for p in active_products)
            model.Add(total_crews_in_shift <= 3)
            
            # Activar variable de penalización si se usan más de 2
            model.Add(total_crews_in_shift <= 2 + three_crews_used[d, s])

    # Cumplimiento Demanda
    for p in active_products:
        rate_turno = get_rate(p, 'turno')
        rate_admin = get_rate(p, 'admin')
        
        total_production = sum(x[d, s, p] * rate_turno for d in range(len(DAYS)) for s in range(3)) + \
                           sum(y[d, p] * rate_admin for d in range(len(DAYS)))
        model.Add(total_production >= demand_dict[p])

    # Objetivos
    obj_terms = []
    time_weight_base = 10
    for d in range(len(DAYS)):
        day_cost = (d + 1) * time_weight_base * 3
        
        # Costo Admin
        for p in active_products:
            priority_discount = 5 if 'Celulosa' in p else 0
            obj_terms.append(y[d, p] * (day_cost - priority_discount)) 

        # Costo Turnos
        for s in range(3):
            shift_cost = (s + 1) * time_weight_base
            total_time_cost = day_cost + shift_cost
            
            for p in active_products:
                priority_discount = 5 if 'Celulosa' in p else 0
                obj_terms.append(x[d, s, p] * (total_time_cost - priority_discount)) 
            
            # Penalización fuerte por usar 3 cuadrillas
            obj_terms.append(three_crews_used[d, s] * 500) 

    model.Minimize(sum(obj_terms))

    # Resolver
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        results = []
        shifts_labels = ['T1', 'T2', 'T3']
        for d in range(len(DAYS)):
            # Output Admin
            for p in active_products:
                if solver.Value(y[d, p]) > 0:
                    prod = get_rate(p, 'admin')
                    # Separamos Cliente y Producto para la tabla
                    base_prod, client = p.split('_')
                    results.append([DAYS[d], 'Admin', client, base_prod, 1, prod])
            # Output Turnos
            for s in range(3):
                for p in active_products:
                    num_crews = solver.Value(x[d, s, p])
                    if num_crews > 0:
                        prod = num_crews * get_rate(p, 'turno')
                        base_prod, client = p.split('_')
                        results.append([DAYS[d], shifts_labels[s], client, base_prod, num_crews, prod])
        
        df_results = pd.DataFrame(results, columns=['Día', 'Turno', 'Cliente', 'Producto', 'Cuadrillas', 'Producción'])
        return df_results, status
    else:
        return None, status

# --- 4. EJECUCIÓN Y VISUALIZACIÓN ---

if run_optimization:
    total_demand = sum(demands.values())
    
    with st.spinner("Optimizando turnos por cliente..."):
        df_plan, status = solve_schedule(demands)

    if df_plan is not None:
        # --- TRANSFORMACIÓN PARA TABLA ---
        
        # 1. Formato de Celda HTML
        df_plan['Info_Celda'] = df_plan.apply(
            lambda row: f"<div class='cell-content'><div class='crew-badge'>👥 {row['Cuadrillas']}</div><div class='prod-info'>📦 {int(row['Producción'])}</div></div>", axis=1
        )

        # 2. Pivot Table con Índice Múltiple (Cliente, Producto)
        pivot_excel = df_plan.pivot_table(
            index=['Cliente', 'Producto'],
            columns=['Día', 'Turno'],
            values='Info_Celda',
            aggfunc=lambda x: ''.join(x) 
        )

        # 3. Reindexar columnas (Orden correcto)
        full_columns = pd.MultiIndex.from_product([DAYS, SHIFTS_ORDER], names=['Día', 'Turno'])
        pivot_excel = pivot_excel.reindex(columns=full_columns)
        
        # 4. Rellenar vacíos
        pivot_excel = pivot_excel.fillna("<span style='color: #eee;'>-</span>")

        # --- ESTILOS CSS (SCROLLABLE + DISEÑO) ---
        st.markdown("""
        <style>
            /* Contenedor principal para hacer scroll horizontal */
            .table-container {
                width: 100%;
                overflow-x: auto;
                white-space: nowrap;
                border: 1px solid #ddd;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            /* Estilos de la tabla */
            .styled-table {
                width: 100%;
                border-collapse: collapse;
                font-family: sans-serif;
                font-size: 0.9em;
            }
            
            /* Celdas y Encabezados */
            .styled-table th, .styled-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
                min-width: 80px; /* Ancho mínimo para forzar el scroll si es necesario */
            }
            
            /* Encabezado Días (Azul oscuro) */
            .styled-table thead tr:first-child th {
                background-color: #004a7c;
                color: white;
                font-weight: bold;
                text-transform: uppercase;
            }
            
            /* Encabezado Turnos (Gris claro) */
            .styled-table thead tr:nth-child(2) th {
                background-color: #f0f2f6;
                color: #333;
                font-size: 0.8em;
            }
            
            /* Columnas fijas a la izquierda (Cliente/Producto) - Opcional sticky */
            .styled-table tbody th {
                background-color: #fafafa;
                font-weight: bold;
                text-align: left;
                padding-left: 10px;
            }

            /* Estilo interno de celda */
            .cell-content {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 2px;
            }
            .crew-badge {
                background-color: #e3f2fd;
                color: #0d47a1;
                padding: 2px 6px;
                border-radius: 10px;
                font-weight: bold;
                font-size: 0.85em;
            }
            .prod-info {
                color: #555;
                font-size: 0.8em;
            }
        </style>
        """, unsafe_allow_html=True)

        st.success(f"Plan Generado. Total: {total_demand} u.")
        
        # Renderizamos la tabla dentro del div contenedor
        html = pivot_excel.to_html(escape=False, classes="styled-table")
        st.markdown(f'<div class="table-container">{html}</div>', unsafe_allow_html=True)

        # --- VALIDACIÓN POR CLIENTE ---
        st.markdown("### Resumen de Cumplimiento")
        val_df = df_plan.groupby(['Cliente', 'Producto'])['Producción'].sum().reset_index()
        # Mapeamos la meta original
        val_df['Meta'] = val_df.apply(lambda x: demands[f"{x['Producto']}_{x['Cliente']}"], axis=1)
        val_df['%'] = (val_df['Producción'] / val_df['Meta'] * 100).fillna(0).round(1)
        st.dataframe(val_df, use_container_width=True)

    elif status == cp_model.INFEASIBLE:
        st.error("Imposible cumplir la demanda total con las restricciones de cuadrillas (Máx 3 por turno entre todos los clientes).")
    else:
        st.error("Error al optimizar.")