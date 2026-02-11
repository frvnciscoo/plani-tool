import streamlit as st
import pandas as pd
import plotly.express as px
from ortools.sat.python import cp_model
from datetime import datetime, timedelta, time

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Planificador CFS Multicliente")

# Rendimientos (Cnts por turno)
RENDIMIENTOS = {
    "Plywood":            {"Turno": 13, "Admin": 17},
    "Moldura":            {"Turno": 15, "Admin": 20},
    "Mad Seca (MAS)":     {"Turno": 15, "Admin": 20},
    "Mad Verde (MAV)":    {"Turno": 20, "Admin": 26},
    "Celulosa":           {"Turno": 35, "Admin": 46},
    "Papel Kraft":        {"Turno": 10, "Admin": 13}
}

TURNOS_INFO = {0: "T1", 1: "T2", 2: "T3"} 
TURNOS_MAP_REVERSE = {"T1": 0, "T2": 1, "T3": 2} 

# Estado de sesión
if 'naves_db' not in st.session_state:
    st.session_state['naves_db'] = {} 

# ==============================================================================
# 2. FUNCIONES AUXILIARES
# ==============================================================================
def obtener_turno_de_hora(hora):
    h_val = hora.hour + (hora.minute / 60.0)
    if 8 <= h_val < 15.5: return 0, "T1"
    elif 15.5 <= h_val < 23: return 1, "T2"
    else: return 2, "T3"

# ==============================================================================
# 3. MOTOR DE OPTIMIZACIÓN
# ==============================================================================
def optimizar_plan(naves_db, fecha_inicio_simulacion, turno_inicio_str):
    model = cp_model.CpModel()
    
    if not naves_db:
        return pd.DataFrame(), pd.DataFrame(), "No hay naves registradas."

    # --- A. Preparación de Tiempos ---
    fechas_corte = [data['datetime_corte'].date() for data in naves_db.values()]
    if not fechas_corte:
        return pd.DataFrame(), pd.DataFrame(), "Datos de naves incompletos."

    max_date = max(fechas_corte)
    today = fecha_inicio_simulacion 
    
    # Aumentamos el horizonte (+5 días) para dar espacio a cargas atrasadas
    days_horizon = max((max_date - today).days + 5, 2)
    num_shifts = days_horizon * 3
    
    products = list(RENDIMIENTOS.keys())
    
    # Offset de inicio (para bloquear el pasado)
    start_offset = TURNOS_MAP_REVERSE[turno_inicio_str]

    # --- B. Variables ---
    x = {} 
    production = {} 

    for s in range(num_shifts):
        for p in products:
            x[p, s, 'Turno'] = model.NewIntVar(0, 3, f'gang_T_{p}_{s}')
            x[p, s, 'Admin'] = model.NewIntVar(0, 1, f'gang_A_{p}_{s}')
            
            rate_t = RENDIMIENTOS[p]['Turno']
            rate_a = RENDIMIENTOS[p]['Admin']
            
            production[p, s] = model.NewIntVar(0, 2000, f'prod_{p}_{s}')
            model.Add(production[p, s] == x[p, s, 'Turno'] * rate_t + x[p, s, 'Admin'] * rate_a)

    # --- C. Restricciones Operativas ---
    
    for s in range(num_shifts):
        # Bloqueo de pasado
        if s < start_offset:
            for p in products:
                model.Add(x[p, s, 'Turno'] == 0)
                model.Add(x[p, s, 'Admin'] == 0)
        else:
            model.Add(sum(x[p, s, 'Turno'] for p in products) <= 3)
            model.Add(sum(x[p, s, 'Admin'] for p in products) <= 1)
            
            # Regla: Admin SOLO en T1
            if s % 3 != 0:
                for p in products:
                    model.Add(x[p, s, 'Admin'] == 0)

    # Regla: Max 1 Admin global por día
    for d in range(days_horizon):
        s1, s2, s3 = d*3, d*3+1, d*3+2
        current_shifts = [s for s in [s1, s2, s3] if s < num_shifts]
        total_admin_day = sum(x[p, s, 'Admin'] for p in products for s in current_shifts)
        model.Add(total_admin_day <= 1)

    # --- D. Procesamiento de Demandas y Penalizaciones ---
    demandas_detalladas = []
    
    # Pre-procesar demandas
    grouped_demands = [] # Lista de tuplas (producto, cantidad, deadline_idx)
    
    for nombre_nave, datos in naves_db.items():
        dt_corte = datos['datetime_corte']
        diff_days = (dt_corte.date() - today).days
        t_idx, t_name = obtener_turno_de_hora(dt_corte.time())
        
        deadline_raw = (diff_days * 3) + t_idx
        deadline_idx = max(0, deadline_raw)
        
        for item in datos['carga']:
            grouped_demands.append({
                'producto': item['producto'],
                'cantidad': item['cantidad'],
                'deadline': deadline_idx
            })
            demandas_detalladas.append({
                'nave': nombre_nave,
                'cliente': item['cliente'],
                'producto': item['producto'],
                'cantidad': item['cantidad'],
                'deadline': deadline_idx,
                'fecha_corte': dt_corte.date(),
                'hora_corte': dt_corte.time(),
                'turno_corte': t_name
            })

    # Construcción de Restricciones y Costos
    cost = 0
    
    for p in products:
        reqs_prod = [d for d in grouped_demands if d['producto'] == p]
        reqs_prod.sort(key=lambda r: r['deadline'])
        
        acumulado = 0
        for r in reqs_prod:
            acumulado += r['cantidad']
            deadline = r['deadline']
            
            # 1. RESTRICCIÓN FINAL: Al acabar la simulación, DEBE estar hecho (No negociable)
            total_horizon_prod = sum(production[p, s] for s in range(num_shifts))
            model.Add(total_horizon_prod >= acumulado)
            
            # 2. RESTRICCIÓN SUAVE (SOFT CONSTRAINT) EN EL DEADLINE
            # Permitimos fallar, pero con penalización monstruosa
            
            if deadline < num_shifts:
                # Calculamos producción hasta el deadline
                prod_at_deadline = sum(production[p, s] for s in range(deadline + 1))
                
                # Variable de Holgura (Lo que faltó en el momento del corte)
                shortfall_at_cutoff = model.NewIntVar(0, 100000, f'short_cut_{p}_{deadline}')
                model.Add(prod_at_deadline + shortfall_at_cutoff >= acumulado)
                
                # PENALIZACIÓN 1: EL GOLPE (Multa por no tenerlo listo al corte)
                cost += shortfall_at_cutoff * 1000000 
                
                # PENALIZACIÓN 2: LA CARRERA (Minimizar el retraso)
                # Para cada turno DESPUÉS del deadline, si todavía falta, cobramos multa.
                # Esto obliga a que el 'shortfall' baje a cero lo más rápido posible.
                for s_late in range(deadline + 1, num_shifts):
                    prod_at_late = sum(production[p, s] for s in range(s_late + 1))
                    shortfall_late = model.NewIntVar(0, 100000, f'short_late_{p}_{s_late}')
                    model.Add(prod_at_late + shortfall_late >= acumulado)
                    
                    # Multa por cada turno que pasa y sigo debiendo carga
                    cost += shortfall_late * 5000 

            # 3. Restricción Suave "1 Día Antes" (Buffer) - Prioridad Menor
            soft_deadline = deadline - 3
            if soft_deadline >= start_offset:
                prod_soft = sum(production[p, s] for s in range(min(soft_deadline, num_shifts - 1) + 1))
                shortfall_buffer = model.NewIntVar(0, 100000, f'short_buff_{p}_{deadline}')
                model.Add(prod_soft + shortfall_buffer >= acumulado)
                cost += shortfall_buffer * 50 # Peso bajo comparado con fallar el deadline

    # Costo Operativo (Minimizar recursos usados)
    for s in range(num_shifts):
        if s >= start_offset:
            t_gangs = sum(x[p, s, 'Turno'] for p in products)
            a_gangs = sum(x[p, s, 'Admin'] for p in products)
            
            is_saturated = model.NewBoolVar(f'sat_{s}')
            model.Add(t_gangs >= 3).OnlyEnforceIf(is_saturated)
            model.Add(t_gangs < 3).OnlyEnforceIf(is_saturated.Not())
            
            # Peso muy bajo (10-100) vs Peso de Falla (1,000,000)
            cost += t_gangs*10 + (a_gangs) + (is_saturated * 100)

    model.Minimize(cost)
    
    # --- E. Solución ---
    solver = cp_model.CpSolver()
    # Aumentamos un poco el tiempo límite por si es complejo
    solver.parameters.max_time_in_seconds = 10.0 
    status = solver.Solve(model)
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        pool_produccion = []
        for s in range(num_shifts):
            if s >= start_offset:
                for p in products:
                    val = solver.Value(production[p, s])
                    if val > 0:
                        day_idx = s // 3
                        shift_idx = s % 3
                        fecha_real = today + timedelta(days=day_idx)
                        
                        pool_produccion.append({
                            's_idx': s,
                            'fecha': fecha_real,
                            'turno_nom': TURNOS_INFO[shift_idx],
                            'producto': p,
                            'cantidad_disponible': val,
                            'c_turno': solver.Value(x[p, s, 'Turno']),
                            'c_admin': solver.Value(x[p, s, 'Admin'])
                        })

        demandas_detalladas.sort(key=lambda x: x['deadline'])
        plan_asignado = []
        
        # Asignación FIFO inteligente
        # Nota: Aquí no forzamos deadline en el IF, permitimos tomar slots futuros
        for demanda in demandas_detalladas:
            qty_needed = demanda['cantidad']
            prod_p = demanda['producto']
            deadline_nave = demanda['deadline']
            
            for slot in pool_produccion:
                # Aceptamos cualquier slot válido (incluso si es post-deadline, porque ya pagamos la multa)
                if slot['producto'] == prod_p and slot['cantidad_disponible'] > 0:
                    
                    tomar = min(qty_needed, slot['cantidad_disponible'])
                    
                    # Detectar si está atrasado para marcarlo visualmente si quisiéramos
                    es_tardio = slot['s_idx'] > deadline_nave
                    
                    plan_asignado.append({
                        'Nave': demanda['nave'],
                        'Cliente': demanda['cliente'],
                        'Producto': demanda['producto'],
                        'Fecha': slot['fecha'],
                        'Turno': slot['turno_nom'],
                        'Cantidad': tomar,
                        'CutOff_Date': demanda['fecha_corte'], 
                        'CutOff_Turn': demanda['turno_corte'],
                        'Estado': 'ATRASADO' if es_tardio else 'OK' # Dato extra útil
                    })
                    slot['cantidad_disponible'] -= tomar
                    qty_needed -= tomar
                    if qty_needed <= 0: break
        
        df_recursos = pd.DataFrame(pool_produccion)
        if not df_recursos.empty:
            df_recursos = df_recursos.groupby(['fecha', 'turno_nom', 'producto'])[['c_turno', 'c_admin']].max().reset_index()
            df_recursos['Periodo'] = df_recursos['fecha'].astype(str) + " " + df_recursos['turno_nom']

        df_matrix = pd.DataFrame(plan_asignado)
        
        # Mensaje personalizado según si hubo atrasos
        if any(d['Estado'] == 'ATRASADO' for d in plan_asignado):
            msg = "Plan Generado con EXCEPCIONES: Alguna carga quedó fuera de fecha (revisar columnas post-cierre)."
        else:
            msg = "Plan Generado Exitosamente (Todo en fecha)."

        return df_matrix, df_recursos, msg
    else:
        return pd.DataFrame(), pd.DataFrame(), "No es factible. Revisa si la capacidad total del horizonte alcanza."

# ==============================================================================
# 4. INTERFAZ
# ==============================================================================
st.title("⚓ Planificador CFS - Multi Cliente")

with st.sidebar:
    st.header("⚙️ Configuración")
    col_conf1, col_conf2 = st.columns(2)
    fecha_inicio_plan = col_conf1.date_input("Fecha Inicio", value=datetime.now().date())
    turno_inicio_plan = col_conf2.selectbox("Turno Inicio", ["T1", "T2", "T3"])
    st.caption(f"Inicio: {fecha_inicio_plan} ({turno_inicio_plan})")
    st.divider()

    st.header("1. Gestión de Naves")
    with st.expander("🚢 Añadir Nueva Nave", expanded=True):
        new_nave_name = st.text_input("Nombre Nave")
        col_d, col_t = st.columns(2)
        d_input = col_d.date_input("Fecha Corte", value=datetime.now())
        t_input = col_t.time_input("Hora Corte", value=time(15, 30))
        
        if st.button("Crear Nave"):
            if new_nave_name and new_nave_name not in st.session_state['naves_db']:
                dt_full = datetime.combine(d_input, t_input)
                st.session_state['naves_db'][new_nave_name] = {"datetime_corte": dt_full, "carga": []}
                st.success(f"Nave {new_nave_name} creada.")
                st.rerun()
            elif new_nave_name in st.session_state['naves_db']:
                st.warning("Ya existe.")

    if st.session_state['naves_db']:
        st.header("2. Cargar Manifiesto")
        with st.form("add_cargo_form"):
            s_nave = st.selectbox("Seleccionar Nave", list(st.session_state['naves_db'].keys()))
            c1, c2 = st.columns(2)
            s_cliente = c1.selectbox("Cliente", ["ARAUCO", "CMPC"])
            s_prod = c2.selectbox("Producto", list(RENDIMIENTOS.keys()))
            s_qty = st.number_input("Cantidad (Cnts)", min_value=1, step=1)
            if st.form_submit_button("➕ Agregar"):
                st.session_state['naves_db'][s_nave]['carga'].append({"cliente": s_cliente, "producto": s_prod, "cantidad": s_qty})
                st.success(f"Agregado a {s_nave}")
                st.rerun()
    st.divider()
    
    st.subheader("📋 Naves Activas")
    naves_to_delete = []
    for nave, data in st.session_state['naves_db'].copy().items():
        if 'datetime_corte' not in data:
            naves_to_delete.append(nave)
            continue
        dt_show = data['datetime_corte']
        with st.expander(f"{nave} | {dt_show.strftime('%d-%m %H:%M')}"): 
            if not data['carga']: st.write("*Sin carga*")
            else:
                for item in data['carga']: st.write(f"- {item['cantidad']} {item['producto']}")
            if st.button(f"Borrar {nave}", key=f"del_{nave}"): naves_to_delete.append(nave)
    if naves_to_delete:
        for n in set(naves_to_delete): 
            if n in st.session_state['naves_db']: del st.session_state['naves_db'][n]
        st.rerun()

# ==============================================================================
# 5. DASHBOARD PRINCIPAL
# ==============================================================================

if st.session_state['naves_db']:
    if st.button("🚀 Calcular Planificación Óptima", type="primary"):
        df_matrix, df_recursos, msg = optimizar_plan(st.session_state['naves_db'], fecha_inicio_plan, turno_inicio_plan)
        
        if not df_matrix.empty:
            if "EXCEPCIONES" in msg: st.warning(f"⚠️ {msg}")
            else: st.success(f"✅ {msg}")
            
            st.subheader("📋 Matriz de Planificación")
            
            pivot_df = df_matrix.pivot_table(
                index=['Nave', 'Cliente', 'Producto', 'CutOff_Date', 'CutOff_Turn'],
                columns=['Fecha', 'Turno'],
                values='Cantidad',
                aggfunc='sum',
                fill_value=0
            )

            # Reindexar para mostrar todo el rango
            min_date = df_matrix['Fecha'].min()
            max_date = df_matrix['Fecha'].max()
            days_span = (max_date - min_date).days + 1
            full_columns = []
            for d in range(days_span):
                current_date = min_date + timedelta(days=d)
                for t in ["T1", "T2", "T3"]: 
                    full_columns.append((current_date, t))
            full_index = pd.MultiIndex.from_tuples(full_columns, names=['Fecha', 'Turno'])
            pivot_df = pivot_df.reindex(columns=full_index, fill_value=0)            
            
            def format_zero(val): return "" if val == 0 else f"{val:.0f}"

            def highlight_cutoff_precise(row):
                if len(row.name) > 4:
                    cutoff_date = row.name[3]
                    cutoff_turn = row.name[4]
                else: return ['' for _ in row]
                
                styles = []
                # Flag para saber si ya pasamos el cutoff
                pasado_cutoff = False
                
                for (col_date, col_turno), val in row.items():
                    # Lógica estricta de orden temporal
                    dt_col = datetime.combine(col_date, time(0,0)) # Dummy time
                    dt_cut = datetime.combine(cutoff_date, time(0,0))
                    
                    # Comparar turno es complicado sin mapa, pero usaremos igualdad exacta para el borde rojo
                    es_cutoff = (col_date == cutoff_date and col_turno == cutoff_turn)
                    
                    if es_cutoff:
                        styles.append('background-color: #ffe6e6; color: #990000; border: 2px solid #ff4b4b !important; font-weight: bold;')
                    elif val > 0:
                        # Si hay valor y estamos en fecha posterior O (misma fecha y turno posterior? aprox)
                        if col_date > cutoff_date:
                             styles.append('background-color: #fff4e5; color: #d97706; font-weight: bold;') # Naranja: Atrasado
                        else:
                             styles.append('')
                    else:
                        styles.append('')
                return styles

            w_nave, w_clie, w_prod = 100, 100, 140
            st.markdown(f"""
                <style>
                .table-container {{ width: 100%; max-height: 600px; overflow: auto; border: 1px solid #ccc; position: relative; }}
                .custom-table {{ border-collapse: separate; border-spacing: 0; font-family: sans-serif; font-size: 13px; width: max-content; background-color: white; }}
                .custom-table th, .custom-table td {{ border-right: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0; padding: 6px 10px; white-space: nowrap; }}
                .custom-table thead tr th {{ position: sticky; top: 0; background-color: #f0f2f6 !important; color: #333; z-index: 10; border-top: 1px solid #ccc; border-bottom: 2px solid #bbb; }}
                .custom-table thead tr:nth-child(2) th {{ top: 30px; z-index: 10; }}
                .custom-table tbody th.level0 {{ position: sticky; left: 0; min-width: {w_nave}px; max-width: {w_nave}px; background-color: #fff !important; z-index: 9; }}
                .custom-table tbody th.level1 {{ position: sticky; left: {w_nave}px; min-width: {w_clie}px; max-width: {w_clie}px; background-color: #fff !important; z-index: 9; }}
                .custom-table tbody th.level2 {{ position: sticky; left: {w_nave + w_clie}px; min-width: {w_prod}px; max-width: {w_prod}px; background-color: #fff !important; z-index: 9; border-right: 2px solid #aaa !important; box-shadow: 4px 0 5px -2px rgba(0,0,0,0.1); }}
                .custom-table thead th:nth-child(1) {{ position: sticky !important; left: 0 !important; z-index: 20 !important; background-color: #e6e9ef !important; min-width: {w_nave}px; max-width: {w_nave}px; }}
                .custom-table thead th:nth-child(2) {{ position: sticky !important; left: {w_nave}px !important; z-index: 20 !important; background-color: #e6e9ef !important; min-width: {w_clie}px; max-width: {w_clie}px; }}
                .custom-table thead th:nth-child(3) {{ position: sticky !important; left: {w_nave + w_clie}px !important; z-index: 20 !important; background-color: #e6e9ef !important; border-right: 2px solid #aaa !important; min-width: {w_prod}px; max-width: {w_prod}px; }}
                </style>
            """, unsafe_allow_html=True)

            html_table = pivot_df.style\
                .format(format_zero)\
                .apply(highlight_cutoff_precise, axis=1)\
                .hide(level='CutOff_Date', axis=0) \
                .hide(level='CutOff_Turn', axis=0) \
                .set_table_attributes('class="custom-table"')\
                .to_html(escape=False, sparsify=False)
            
            st.markdown(f'<div class="table-container">{html_table}</div>', unsafe_allow_html=True)
            
            st.divider()


            
        else:
            st.error(f"❌ {msg}")
else:
    st.info("👈 Agrega Naves para comenzar.")
