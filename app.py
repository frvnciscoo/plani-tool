import streamlit as st
import pandas as pd
import plotly.express as px
from ortools.sat.python import cp_model
from datetime import datetime, timedelta, time
import re

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
st.set_page_config(layout="wide", page_title="Planificador CFS Multicliente")

# Rendimientos (Cnts por turno)
RENDIMIENTOS = {
    "Plywood":            {"Turno": 13, "Admin": 17},
    "Molduras":           {"Turno": 15, "Admin": 20},
    "Madera Seca (MAS)":  {"Turno": 15, "Admin": 20},
    "Madera Verde (MAV)": {"Turno": 20, "Admin": 26},
    "Celulosa":           {"Turno": 35, "Admin": 46},
    "Papel Kraft":        {"Turno": 10, "Admin": 13},
    "Otros":              {"Turno": 10, "Admin": 12}
}

TURNOS_INFO = {0: "T1", 1: "T2", 2: "T3"} 
TURNOS_MAP_REVERSE = {"T1": 0, "T2": 1, "T3": 2} 

# Estado de sesión
if 'naves_db' not in st.session_state:
    st.session_state['naves_db'] = {} 
if 'limites_turno' not in st.session_state:
    st.session_state['limites_turno'] = {}

# ==============================================================================
# 2. FUNCIONES AUXILIARES
# ==============================================================================
def obtener_turno_de_hora(hora):
    h_val = hora.hour + (hora.minute / 60.0)
    if 8 <= h_val < 15.5: return 0, "T1"
    elif 15.5 <= h_val < 23: return 1, "T2"
    else: return 2, "T3"

def extraer_fecha_cutoff(texto_stacking):
    if not isinstance(texto_stacking, str): return None
    match = re.search(r"(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2})", texto_stacking)
    if match:
        try: return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%d/%m/%Y %H:%M")
        except ValueError: return None
    return None

def homologar_cliente(val):
    v = str(val).upper().strip()
    if v.startswith("CMPC"): return "CMPC"
    if v.startswith("ARAUCO"): return "ARAUCO"
    return v

def homologar_producto(producto_raw):
    t = str(producto_raw).upper().strip()
    identificadores_celulosa = ["CELULOSA", "CEL DP", "CEL BKP", "CEL EKP"]
    if any(id_cel in t for id_cel in identificadores_celulosa): return "Celulosa"
    if "PAPEL KRAFT" in t: return "Papel Kraft"
    if "PLYWOOD" in t: return "Plywood"
    if "MDF" in t or "TRUPAN" in t: return "Molduras"
    lista_molduras = ["CLEARS", "AGLOMERADOS", "TABLERO", "OSB", "CHAPAS", "MOLDURAS", "MOLDING"]
    if any(item in t for item in lista_molduras): return "Molduras"
    lista_mav = ["BLANKS", "MPALLVERDE", "BASAS", "M.ASER.VERDE", "VERDE", "MAV"]
    if any(item in t for item in lista_mav): return "Madera Verde (MAV)"
    lista_mas = ["SHOP", "MOULDINGBETTER", "MPALLSECA", "M.ASER. SECA", "SECA", "MAS"]
    if any(item in t for item in lista_mas): return "Madera Seca (MAS)"
    return "Otros"

def procesar_archivo_carga(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
        else: df = pd.read_excel(uploaded_file)
        
        df.columns = df.columns.str.strip()
        required_cols = ['Stacking', 'Nave', 'Cliente', 'Producto', 'Box Saldo', 'Terminal']
        if not all(col in df.columns for col in required_cols):
            return None, None, f"Faltan columnas clave: {required_cols}"

        # Filtrar por Terminal
        df = df[df['Terminal'].astype(str).str.upper().str.strip() == "SVTI"].copy()
        
        # ---> NUEVO: Filtrar exclusiones de ROLEO <---
        if 'N° Reserva' in df.columns:
            df = df[~df['N° Reserva'].astype(str).str.upper().str.contains("ROLEO", na=False)]
            
        if df.empty: return None, None, "No hay registros SVTI válidos o todos son ROLEO."
        
        df['dt_corte'] = df['Stacking'].astype(str).apply(extraer_fecha_cutoff)
        df_fechas = df.dropna(subset=['dt_corte']).groupby('Nave')['dt_corte'].min().reset_index()
        dict_fechas = pd.Series(df_fechas.dt_corte.values, index=df_fechas.Nave).to_dict()

        df['Cliente_Homologado'] = df['Cliente'].apply(homologar_cliente)
        df['Producto_Homologado'] = df['Producto'].apply(homologar_producto)
        df = df[df['Box Saldo'] > 0]
        
        df_resumen = df.groupby(['Nave', 'Cliente_Homologado', 'Producto_Homologado'])['Box Saldo'].sum().reset_index()
        df_resumen = df_resumen.rename(columns={'Cliente_Homologado': 'Cliente'})
        
        return df_resumen, dict_fechas, "OK"
    except Exception as e:
        return None, None, str(e)

# ==============================================================================
# 3. MOTOR DE OPTIMIZACIÓN (ACTUALIZADO)
# ==============================================================================
# ### NUEVO: Agregamos parámetros extra a la función
def optimizar_plan(naves_db, fecha_inicio_simulacion, turno_inicio_str, cant_admin_max, consolidar_domingo, cliente_filtro, limites_turno_dinamicos):
    model = cp_model.CpModel()
    
    if not naves_db: return pd.DataFrame(), pd.DataFrame(), "No hay naves."

    fechas_corte = [data['datetime_corte'].date() for data in naves_db.values()]
    if not fechas_corte: return pd.DataFrame(), pd.DataFrame(), "Faltan fechas de corte."

    max_date = max(fechas_corte)
    today = fecha_inicio_simulacion 
    days_horizon = max((max_date - today).days + 7, 2)
    num_shifts = days_horizon * 3
    
    products = list(RENDIMIENTOS.keys())
    start_offset = TURNOS_MAP_REVERSE[turno_inicio_str]

    # --- B. Variables ---
    x = {} 
    production = {} 

    for s in range(num_shifts):
        for p in products:
            # Subimos el límite base a 5 por si pones excepciones altas en la UI
            x[p, s, 'Turno'] = model.NewIntVar(0, 5, f'gang_T_{p}_{s}')
            x[p, s, 'Admin'] = model.NewIntVar(0, cant_admin_max, f'gang_A_{p}_{s}')
            
            rate_t = RENDIMIENTOS[p]['Turno']
            rate_a = RENDIMIENTOS[p]['Admin']
            
            production[p, s] = model.NewIntVar(0, 5000, f'prod_{p}_{s}')
            model.Add(production[p, s] == x[p, s, 'Turno'] * rate_t + x[p, s, 'Admin'] * rate_a)

    # --- C. Restricciones Operativas y de Calendario ---
    for s in range(num_shifts):
        if s < start_offset:
            for p in products:
                model.Add(x[p, s, 'Turno'] == 0)
                model.Add(x[p, s, 'Admin'] == 0)
        else:
            day_idx = s // 3
            fecha_actual = today + timedelta(days=day_idx)
            dia_semana = fecha_actual.weekday() 
            turno_str = TURNOS_INFO[s % 3]
            
            # --- LÍMITE DE CUADRILLAS A TURNO (Leído desde la fila editable UI) ---
            clave_fecha_turno = f"{fecha_actual.strftime('%Y-%m-%d')} {turno_str}"
            limite_max_turno = limites_turno_dinamicos.get(clave_fecha_turno, 2) 
            model.Add(sum(x[p, s, 'Turno'] for p in products) <= limite_max_turno)
            
            # --- LÓGICA DE ADMINISTRATIVAS (Automática) ---
            model.Add(sum(x[p, s, 'Admin'] for p in products) <= cant_admin_max)
            if dia_semana < 5: 
                # LUNES A VIERNES: Solo T1
                if s % 3 != 0:
                    for p in products: model.Add(x[p, s, 'Admin'] == 0)
            else:
                # SÁBADO Y DOMINGO: Cero Admin
                for p in products: model.Add(x[p, s, 'Admin'] == 0)

    # Límite diario de Admins
    for d in range(days_horizon):
        s1, s2, s3 = d*3, d*3+1, d*3+2
        current_shifts = [s for s in [s1, s2, s3] if s < num_shifts]
        total_admin_day = sum(x[p, s, 'Admin'] for p in products for s in current_shifts)
        model.Add(total_admin_day <= cant_admin_max)

    # --- D. Procesamiento de Demandas ---
    demandas_detalladas = []
    grouped_demands = [] 
    
    for nombre_nave, datos in naves_db.items():
        dt_corte = datos['datetime_corte']
        diff_days = (dt_corte.date() - today).days
        t_idx, t_name = obtener_turno_de_hora(dt_corte.time())
        deadline_idx = max(0, (diff_days * 3) + t_idx)
        
            for item in datos['carga']:
                    # ---> APLICAR EL FILTRO DE CLIENTE <---
                    if cliente_filtro != "Todo" and item['cliente'].upper() != cliente_filtro:
                        continue
        
                    if item['producto'] not in products:
                        if "Otros" in products: 
                            item['producto'] = "Otros"
                        else: 
                            continue

            grouped_demands.append({'producto': item['producto'], 'cantidad': item['cantidad'], 'deadline': deadline_idx})
            demandas_detalladas.append({
                'nave': nombre_nave, 'cliente': item['cliente'], 'producto': item['producto'],
                'cantidad': item['cantidad'], 'deadline': deadline_idx,
                'fecha_corte': dt_corte.date(), 'turno_corte': t_name
            })

    cost = 0
    for p in products:
        reqs_prod = [d for d in grouped_demands if d['producto'] == p]
        reqs_prod.sort(key=lambda r: r['deadline'])
        
        acumulado = 0
        for r in reqs_prod:
            acumulado += r['cantidad']
            deadline = r['deadline']
            
            total_horizon_prod = sum(production[p, s] for s in range(num_shifts))
            model.Add(total_horizon_prod >= acumulado)
            
            if deadline < num_shifts:
                prod_at_deadline = sum(production[p, s] for s in range(deadline + 1))
                shortfall_at_cutoff = model.NewIntVar(0, 100000, f'short_cut_{p}_{deadline}')
                model.Add(prod_at_deadline + shortfall_at_cutoff >= acumulado)
                cost += shortfall_at_cutoff * 1000000 
                
                for s_late in range(deadline + 1, num_shifts):
                    prod_at_late = sum(production[p, s] for s in range(s_late + 1))
                    shortfall_late = model.NewIntVar(0, 100000, f'short_late_{p}_{s_late}')
                    model.Add(prod_at_late + shortfall_late >= acumulado)
                    cost += shortfall_late * 5000 

    for s in range(num_shifts):
        if s >= start_offset:
            t_gangs = sum(x[p, s, 'Turno'] for p in products)
            a_gangs = sum(x[p, s, 'Admin'] for p in products)
            
            # --- PENALIZACIÓN DINÁMICA ---
            day_idx = s // 3
            fecha_actual = today + timedelta(days=day_idx)
            turno_str = TURNOS_INFO[s % 3]
            clave_fecha_turno = f"{fecha_actual.strftime('%Y-%m-%d')} {turno_str}"
            limite_s = limites_turno_dinamicos.get(clave_fecha_turno, 2)
            
            if limite_s > 0:
                is_saturated = model.NewBoolVar(f'sat_{s}')
                model.Add(t_gangs >= limite_s).OnlyEnforceIf(is_saturated)
                model.Add(t_gangs < limite_s).OnlyEnforceIf(is_saturated.Not())
                cost += t_gangs*10 + (a_gangs) + (is_saturated * 100)
            else:
                cost += t_gangs*10 + (a_gangs)

    model.Minimize(cost)
    
    # --- E. Solución ---
    solver = cp_model.CpSolver()
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
        
        for demanda in demandas_detalladas:
            qty_needed = demanda['cantidad']
            prod_p = demanda['producto']
            deadline_nave = demanda['deadline']
            
            for slot in pool_produccion:
                if slot['producto'] == prod_p and slot['cantidad_disponible'] > 0:
                    tomar = min(qty_needed, slot['cantidad_disponible'])
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
                        'Estado': 'ATRASADO' if es_tardio else 'OK'
                    })
                    slot['cantidad_disponible'] -= tomar
                    qty_needed -= tomar
                    if qty_needed <= 0: break
        
        df_recursos = pd.DataFrame(pool_produccion)
        if not df_recursos.empty:
            df_recursos = df_recursos.groupby(['fecha', 'turno_nom', 'producto'])[['c_turno', 'c_admin']].max().reset_index()
            df_recursos['Periodo'] = df_recursos['fecha'].astype(str) + " " + df_recursos['turno_nom']

        df_matrix = pd.DataFrame(plan_asignado)
        
        if any(d['Estado'] == 'ATRASADO' for d in plan_asignado):
            msg = "Plan Generado con EXCEPCIONES: Carga atrasada detectada."
        else:
            msg = "Plan Generado Exitosamente (Todo en fecha)."

        return df_matrix, df_recursos, msg
    else:
        return pd.DataFrame(), pd.DataFrame(), "No es factible. Revisa capacidad vs demanda."

# ==============================================================================
# 4. INTERFAZ
# ==============================================================================
st.title("⚓ Planificador CFS - Multi Cliente")

with st.sidebar:
    st.header("⚙️ Configuración")
    col_conf1, col_conf2 = st.columns(2)
    fecha_inicio_plan = col_conf1.date_input("Fecha Inicio", value=datetime.now().date())
    turno_inicio_plan = col_conf2.selectbox("Turno Inicio", ["T1", "T2", "T3"])

    cliente_filtro = st.selectbox("Filtrar Cliente a Planificar", ["Todo", "ARAUCO", "CMPC"])
    
    st.divider()
    
    # ### NUEVO: Controles de Recursos y Restricciones
    st.subheader("👷 Recursos y Restricciones")
    cant_admin_max = st.radio("Cuadrillas Administrativas (Máx)", [0, 1, 2], index=1, horizontal=True)
    consolidar_domingo = st.toggle("¿Consolidar Domingo?", value=False)
    
    if consolidar_domingo:
        st.caption("✅ Domingos HABILITADOS para trabajo.")
    else:
        st.caption("⛔ Domingos BLOQUEADOS.")
    
    st.info("ℹ️ Nota: Las cuadrillas operativas solo funcionan de Lunes a Viernes.")
    
    st.divider()

    # --- PESTAÑAS DE INGRESO ---
    tab_manual, tab_excel = st.tabs(["📝 Manual", "📂 Excel/CSV"])

    with tab_manual:
        st.subheader("1. Gestión de Naves")
        with st.expander("🚢 Añadir Nueva Nave", expanded=False):
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
            st.subheader("2. Cargar Manifiesto")
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

    with tab_excel:
        st.subheader("Carga Masiva (Saldos)")
        uploaded_file = st.file_uploader("Subir Planilla (Excel/CSV)", type=['xlsx', 'xls', 'csv'])
        if uploaded_file:
            df_resumen, dict_fechas, status = procesar_archivo_carga(uploaded_file)
            if df_resumen is not None:
                st.info("Vista previa de carga:")
                st.dataframe(df_resumen.head())
                if st.button("💾 Procesar y Cargar a Base de Datos"):
                    count_naves = 0
                    for nave, fecha_dt in dict_fechas.items():
                        if nave not in st.session_state['naves_db']:
                            st.session_state['naves_db'][nave] = {"datetime_corte": fecha_dt, "carga": []}
                            count_naves += 1
                        else:
                            if fecha_dt < st.session_state['naves_db'][nave]['datetime_corte']:
                                st.session_state['naves_db'][nave]['datetime_corte'] = fecha_dt
                    
                    count_carga = 0
                    for index, row in df_resumen.iterrows():
                        nave = row['Nave']
                        prod = row['Producto_Homologado']
                    
                        # 🔒 GARANTÍA DE EXISTENCIA
                        if nave not in st.session_state['naves_db']:
                            st.session_state['naves_db'][nave] = {
                                "datetime_corte": datetime.now(),
                                "carga": []
                            }
                    
                        if prod not in RENDIMIENTOS:
                            prod = "Otros"
                    
                        st.session_state['naves_db'][nave]['carga'].append({
                            "cliente": row['Cliente'],
                            "producto": prod,
                            "cantidad": int(row['Box Saldo'])
                        })

    st.divider()
    st.subheader("📋 Naves Activas")
    naves_to_delete = []
    
    for nave, data in st.session_state['naves_db'].copy().items():
        if 'datetime_corte' not in data:
            naves_to_delete.append(nave)
            continue
            
        dt_show = data['datetime_corte']
        
        with st.expander(f"🚢 {nave} | {dt_show.strftime('%d-%m %H:%M')}"): 
            
            # --- NUEVO: Editar Fecha/Hora de Corte ---
            st.caption("✏️ Modificar CutOff / Stacking")
            col_d, col_t = st.columns(2)
            new_date = col_d.date_input("Fecha", value=dt_show.date(), key=f"edit_d_{nave}")
            new_time = col_t.time_input("Hora", value=dt_show.time(), key=f"edit_t_{nave}")
            
            if st.button("💾 Actualizar Corte", key=f"save_{nave}", use_container_width=True):
                st.session_state['naves_db'][nave]['datetime_corte'] = datetime.combine(new_date, new_time)
                st.rerun() # Recarga para mostrar la nueva fecha en el título del expander
                
            st.divider()
            # ----------------------------------------
            
            # Mostrar la carga
            if not data['carga']: 
                st.write("*Sin carga*")
            else:
                df_temp = pd.DataFrame(data['carga'])
                if not df_temp.empty:
                    df_g = df_temp.groupby('producto')['cantidad'].sum()
                    for p, q in df_g.items(): 
                        st.write(f"- {q} {p}")
            
            # Botón de borrar con un toque visual
            if st.button(f"🗑️ Borrar Nave", key=f"del_{nave}"): 
                naves_to_delete.append(nave)
                
    if naves_to_delete:
        for n in set(naves_to_delete): 
            if n in st.session_state['naves_db']: 
                del st.session_state['naves_db'][n]
        st.rerun()

# ==============================================================================
# 5. DASHBOARD PRINCIPAL
# ==============================================================================
if st.session_state['naves_db']:
    
    # --- 1. PRE-CÁLCULO DEL HORIZONTE DE TIEMPO ---
    fechas_corte_pre = [data['datetime_corte'].date() for data in st.session_state['naves_db'].values()]
    max_date_pre = max(fechas_corte_pre) if fechas_corte_pre else fecha_inicio_plan
    days_horizon_pre = max((max_date_pre - fecha_inicio_plan).days + 7, 2)
    
    diccionario_limites = {}
    columnas_ordenadas = []
    
    # Llenamos la fila con la lógica base
    for d in range(days_horizon_pre):
        fecha_obj = fecha_inicio_plan + timedelta(days=d)
        dia_sem = fecha_obj.weekday()
        for t_idx, t_name in TURNOS_INFO.items():
            # Clave única para el diccionario (usada en el modelo)
            clave_interna = f"{fecha_obj.strftime('%Y-%m-%d')} {t_name}"
            # Clave visible para las columnas de la UI
            clave_columna = f"{fecha_obj.strftime('%d-%m')} {t_name}" 
            
            columnas_ordenadas.append(clave_columna)
            
            if dia_sem < 5: 
                diccionario_limites[clave_columna] = 0 if t_idx == 0 else 2
            elif dia_sem == 5: 
                diccionario_limites[clave_columna] = 2
            else: 
                diccionario_limites[clave_columna] = 2 if consolidar_domingo else 0
                
            # Sobreescribir si el usuario ya había editado este límite en la sesión
            if clave_interna in st.session_state['limites_turno']:
                diccionario_limites[clave_columna] = st.session_state['limites_turno'][clave_interna]

    # --- 2. RENDERIZAR LA FILA DE CONTROL EDITABLE ---
    st.subheader("🗓️ Planificación Operativa")
    st.caption("Edita la primera fila para ajustar el máximo de Cuadrillas de Turno. La matriz se recalculará automáticamente.")
    
    # Creamos el DataFrame para la tabla editable
    df_limites = pd.DataFrame([diccionario_limites], index=["Máx Cuadrillas"]).reindex(columns=columnas_ordenadas)
    
    # Configuramos las columnas para que sean numéricas y estrechas
    config_columnas = {col: st.column_config.NumberColumn(col, width="small", min_value=0, max_value=5, step=1) for col in columnas_ordenadas}
    
    # Mostrar la tabla editable
    df_editado = st.data_editor(
        df_limites, 
        use_container_width=True, 
        column_config=config_columnas,
        key="editor_limites" # Clave importante para que Streamlit detecte los cambios
    )
    
    # Guardamos los cambios detectados en la sesión (usando la clave interna que espera el modelo)
    for col in columnas_ordenadas:
        # Reconstruir la clave interna a partir del nombre de la columna visual
        partes = col.split(" ")
        fecha_texto = partes[0]
        turno_texto = partes[1]
        
        # Necesitamos reconstruir el año para la clave interna. 
        # Asumimos que el año es el mismo que fecha_inicio_plan, a menos que el mes cruce de año.
        # (Esto es una simplificación, pero funciona bien para horizontes cortos).
        fecha_dt = datetime.strptime(f"{fecha_inicio_plan.year}-{fecha_texto}", "%Y-%d-%m")
        if fecha_dt.date() < fecha_inicio_plan:
             fecha_dt = datetime.strptime(f"{fecha_inicio_plan.year + 1}-{fecha_texto}", "%Y-%d-%m")
             
        clave_interna = f"{fecha_dt.strftime('%Y-%m-%d')} {turno_texto}"
        
        st.session_state['limites_turno'][clave_interna] = df_editado.iloc[0][col]

    st.divider()

    # --- 3. CÁLCULO Y MATRIZ (Automático) ---
    # Ya no hay botón, se calcula automáticamente con los límites guardados en sesión
    with st.spinner('Optimizando planificación...'):
        df_matrix, df_recursos, msg = optimizar_plan(
            st.session_state['naves_db'], 
            fecha_inicio_plan, 
            turno_inicio_plan,
            cant_admin_max,
            consolidar_domingo,
            cliente_filtro,
            st.session_state['limites_turno']
        )
        
    if not df_matrix.empty:
        if "EXCEPCIONES" in msg: st.warning(f"⚠️ {msg}")
        else: st.success(f"✅ {msg}")
        
        # Preparar la matriz pivotada
        pivot_df = df_matrix.pivot_table(
            index=['Nave', 'Cliente', 'Producto', 'CutOff_Date', 'CutOff_Turn'],
            columns=['Fecha', 'Turno'],
            values='Cantidad',
            aggfunc='sum',
            fill_value=0
        )

        # Asegurar que todas las columnas de tiempo existan
        min_date = df_matrix['Fecha'].min()
        max_date = df_matrix['Fecha'].max()
        days_span = (max_date - min_date).days + 1
        full_columns = []
        for d in range(days_span):
            current_date = min_date + timedelta(days=d)
            for t in ["T1", "T2", "T3"]: full_columns.append((current_date, t))
        full_index = pd.MultiIndex.from_tuples(full_columns, names=['Fecha', 'Turno'])
        pivot_df = pivot_df.reindex(columns=full_index, fill_value=0)            
        
        # Funciones de estilo
        def format_zero(val): return "" if val == 0 else f"{val:.0f}"
        
        def highlight_cutoff_precise(row):
            if len(row.name) > 4:
                cutoff_date = row.name[3]
                cutoff_turn = row.name[4]
            else: return ['' for _ in row]
            styles = []
            for (col_date, col_turno), val in row.items():
                es_cutoff = (col_date == cutoff_date and col_turno == cutoff_turn)
                if es_cutoff: styles.append('background-color: #ffe6e6; color: #990000; border: 2px solid #ff4b4b !important; font-weight: bold;')
                elif val > 0:
                    if col_date > cutoff_date: styles.append('background-color: #fff4e5; color: #d97706; font-weight: bold;')
                    else: styles.append('')
                else: styles.append('')
            return styles

        # Renderizar la tabla HTML
        w_nave, w_clie, w_prod = 100, 100, 140
        st.markdown(f"""
            <style>
            .table-container {{ width: 100%; max-height: 600px; overflow: auto; border: 1px solid #ccc; position: relative; margin-top: -15px; }} /* Margen negativo para acercarla a la tabla editable */
            .custom-table {{ border-collapse: separate; border-spacing: 0; font-family: sans-serif; font-size: 13px; width: max-content; background-color: white; }}
            .custom-table th, .custom-table td {{ border-right: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0; padding: 6px 10px; white-space: nowrap; text-align: center; }}
            .custom-table thead tr th {{ position: sticky; top: 0; background-color: #f0f2f6 !important; color: #333; z-index: 10; border-top: 1px solid #ccc; border-bottom: 2px solid #bbb; }}
            .custom-table thead tr:nth-child(2) th {{ top: 30px; z-index: 10; }}
            .custom-table tbody th.level0 {{ position: sticky; left: 0; min-width: {w_nave}px; max-width: {w_nave}px; background-color: #fff !important; z-index: 9; text-align: left; }}
            .custom-table tbody th.level1 {{ position: sticky; left: {w_nave}px; min-width: {w_clie}px; max-width: {w_clie}px; background-color: #fff !important; z-index: 9; text-align: left; }}
            .custom-table tbody th.level2 {{ position: sticky; left: {w_nave + w_clie}px; min-width: {w_prod}px; max-width: {w_prod}px; background-color: #fff !important; z-index: 9; border-right: 2px solid #aaa !important; box-shadow: 4px 0 5px -2px rgba(0,0,0,0.1); text-align: left; }}
            </style>
        """, unsafe_allow_html=True)

        # Modificamos los encabezados de la tabla HTML para que coincidan visualmente con la editable
        html_table = pivot_df.rename(columns=lambda x: f"{x.strftime('%d-%m')}" if isinstance(x, datetime) else x, level=0).style\
            .format(format_zero)\
            .apply(highlight_cutoff_precise, axis=1)\
            .hide(level='CutOff_Date', axis=0) \
            .hide(level='CutOff_Turn', axis=0) \
            .set_table_attributes('class="custom-table"')\
            .to_html(escape=False, sparsify=False)
        
        st.markdown(f'<div class="table-container">{html_table}</div>', unsafe_allow_html=True)
        
        # --- 4. GRÁFICOS ---
        st.divider()
        st.subheader("📊 Uso de Cuadrillas")
        if not df_recursos.empty:
            df_recursos['Día'] = df_recursos['fecha'].astype(str)
            df_g_rec = df_recursos.groupby(['Día', 'turno_nom'])[['c_turno', 'c_admin']].sum().reset_index()
            fig = px.bar(df_g_rec, x='Día', y=['c_turno', 'c_admin'], 
                         title="Cuadrillas Asignadas por Día (Turno vs Admin)",
                         labels={'value': 'Cantidad Cuadrillas', 'variable': 'Tipo'},
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.error(f"❌ {msg}")
else:
    st.info("👈 Agrega Naves en la barra lateral para comenzar la planificación.")











