import streamlit as st
import pandas as pd
from ortools.sat.python import cp_model
import uuid

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Opti-Plan Pro", layout="wide")

# Rendimientos Base
BASE_RATES = {
    'MAV': {'turno': 20, 'admin': 26},
    'MAS': {'turno': 15, 'admin': 20},
    'Celulosa': {'turno': 35, 'admin': 46},
    'Papel': {'turno': 10, 'admin': 13}
}
BASE_PRODUCTS_KEYS = list(BASE_RATES.keys())

DAYS = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
SHIFTS_ORDER = ['Admin', 'T1', 'T2', 'T3']
PRIORITY_WEIGHTS = {1: 1, 2: 2, 3: 5, 4: 15, 5: 100}
SCALE = 10 # Granularidad (10 = 1.0 Cuadrilla)

# --- 2. GESTIÓN DE ESTADO ---
if 'naves_list' not in st.session_state:
    st.session_state.naves_list = []

def add_nave():
    demandas = {}
    # Arauco
    if st.session_state.ar_cel > 0: demandas['Celulosa_ARAUCO'] = st.session_state.ar_cel
    if st.session_state.ar_mav > 0: demandas['MAV_ARAUCO'] = st.session_state.ar_mav
    if st.session_state.ar_mas > 0: demandas['MAS_ARAUCO'] = st.session_state.ar_mas
    if st.session_state.ar_pap > 0: demandas['Papel_ARAUCO'] = st.session_state.ar_pap
    # CMPC
    if st.session_state.cm_cel > 0: demandas['Celulosa_CMPC'] = st.session_state.cm_cel
    if st.session_state.cm_mav > 0: demandas['MAV_CMPC'] = st.session_state.cm_mav
    if st.session_state.cm_mas > 0: demandas['MAS_CMPC'] = st.session_state.cm_mas
    if st.session_state.cm_pap > 0: demandas['Papel_CMPC'] = st.session_state.cm_pap

    new_nave = {
        'id': str(uuid.uuid4())[:8],
        'nombre': st.session_state.temp_nombre,
        'prioridad': st.session_state.temp_prioridad,
        'demandas': demandas
    }
    
    if new_nave['nombre'] and len(demandas) > 0:
        st.session_state.naves_list.append(new_nave)
        st.toast(f"Nave {new_nave['nombre']} agregada", icon="✅")
    else:
        st.toast("Error: Falta nombre o carga.", icon="⚠️")

def delete_nave(nave_id):
    st.session_state.naves_list = [n for n in st.session_state.naves_list if n['id'] != nave_id]

def clear_all():
    st.session_state.naves_list = []

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("🚢 Gestión de Naves")
    with st.expander("➕ Agregar Nueva Nave", expanded=True):
        st.text_input("Nombre de Nave", key="temp_nombre", placeholder="Ej: Hooge")
        st.slider("Prioridad", 1, 5, 3, key="temp_prioridad")
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🌲 ARAUCO**")
            st.number_input("Celulosa", min_value=0, key="ar_cel")
            st.number_input("MAV", min_value=0, key="ar_mav")
            st.number_input("MAS", min_value=0, key="ar_mas")
            st.number_input("Papel", min_value=0, key="ar_pap")
        with c2:
            st.markdown("**📦 CMPC**")
            st.number_input("Celulosa", min_value=0, key="cm_cel")
            st.number_input("MAV", min_value=0, key="cm_mav")
            st.number_input("MAS", min_value=0, key="cm_mas")
            st.number_input("Papel", min_value=0, key="cm_pap")
        st.button("Agregar", on_click=add_nave, type="primary")

    st.divider()
    st.subheader(f"En Cola ({len(st.session_state.naves_list)})")
    if st.session_state.naves_list:
        for nave in st.session_state.naves_list:
            st.info(f"{nave['nombre']} (Prio {nave['prioridad']})")
            if st.button("Borrar", key=f"del_{nave['id']}"):
                delete_nave(nave['id'])
                st.rerun()

    run_opt = st.button("🚀 CALCULAR PLAN", type="primary", disabled=not st.session_state.naves_list)

# --- 4. OPTIMIZACIÓN CON RESTRICCIÓN DE PRODUCTO ÚNICO ---
def solve_multinave(naves):
    model = cp_model.CpModel()
    x = {} # Turnos
    y = {} # Admin
    three_crews_used = {} 

    # 1. Definir Variables
    for d in range(len(DAYS)):
        for s in range(3):
            three_crews_used[d, s] = model.NewBoolVar(f'3crews_{d}_{s}')
        
        for n_idx, nave in enumerate(naves):
            for p_key in nave['demandas'].keys():
                y[d, n_idx, p_key] = model.NewIntVar(0, 1 * SCALE, f'y_{d}_{n_idx}_{p_key}')
                for s in range(3):
                    x[d, s, n_idx, p_key] = model.NewIntVar(0, 3 * SCALE, f'x_{d}_{s}_{n_idx}_{p_key}')

    # 2. Restricciones Generales
    for d in range(len(DAYS)):
        # Admin Global <= 1
        admin_tasks = [y[d, n, p] for n, nave in enumerate(naves) for p in nave['demandas']]
        model.Add(sum(admin_tasks) <= 1 * SCALE)

        for s in range(3):
            # Turno Global <= 3
            shift_tasks = [x[d, s, n, p] for n, nave in enumerate(naves) for p in nave['demandas']]
            total_scaled = sum(shift_tasks)
            model.Add(total_scaled <= 3 * SCALE)
            model.Add(total_scaled <= 2 * SCALE + three_crews_used[d, s] * 1000)

            # --- NUEVA RESTRICCIÓN: INTEGRIDAD DE PRODUCTO ---
            # Para cada tipo base (Celulosa, MAV, etc), la suma total de cuadrillas
            # asignadas (sumando todos los clientes y naves) debe ser ENTERA.
            # Esto evita que 0.5 cuadrilla haga Celulosa y el otro 0.5 haga MAV.
            for base_prod in BASE_PRODUCTS_KEYS:
                # Recolectamos todas las variables de este tipo de producto en este turno
                prod_vars = []
                for n_idx, nave in enumerate(naves):
                    for p_key in nave['demandas']:
                        if base_prod in p_key: # Ej: "Celulosa" in "Celulosa_ARAUCO"
                            prod_vars.append(x[d, s, n_idx, p_key])
                
                if prod_vars:
                    total_prod_capacity = sum(prod_vars)
                    # Variable auxiliar entera (0, 1, 2, 3 cuadrillas completas)
                    crews_integer = model.NewIntVar(0, 3, f'whole_crews_{d}_{s}_{base_prod}')
                    # Obligamos a que la capacidad total sea multiplo exacto de 10 (SCALE)
                    model.Add(total_prod_capacity == crews_integer * SCALE)

    # 3. Demanda
    for n_idx, nave in enumerate(naves):
        for p_key, qty in nave['demandas'].items():
            base_prod = p_key.split('_')[0]
            rt = BASE_RATES[base_prod]['turno']
            ra = BASE_RATES[base_prod]['admin']
            
            prod = sum(x[d, s, n_idx, p_key] for d in range(len(DAYS)) for s in range(3)) * rt + \
                   sum(y[d, n_idx, p_key] for d in range(len(DAYS))) * ra
            
            # Ajuste matemático: Sum(Var) * Rate >= Demand * Scale
            model.Add(prod >= qty * SCALE)

    # 4. Objetivo
    obj_terms = []
    for d in range(len(DAYS)):
        time_cost = (d + 1) * 10
        for n_idx, nave in enumerate(naves):
            w = PRIORITY_WEIGHTS[nave['prioridad']]
            for p_key in nave['demandas']:
                is_cel = 1 if 'Celulosa' in p_key else 0
                obj_terms.append(y[d, n_idx, p_key] * ((time_cost - is_cel) * w))
                for s in range(3):
                    sc = time_cost + (s + 1)
                    obj_terms.append(x[d, s, n_idx, p_key] * ((sc - is_cel) * w))
        for s in range(3):
            obj_terms.append(three_crews_used[d, s] * 5000)

    model.Minimize(sum(obj_terms))
    
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    res = []
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        for d in range(len(DAYS)):
            for n_idx, nave in enumerate(naves):
                for p_key in nave['demandas']:
                    bp, cl = p_key.split('_')
                    # Admin
                    vy = solver.Value(y[d, n_idx, p_key])
                    if vy > 0:
                        res.append({'Día': DAYS[d], 'Turno': 'Admin', 'Nave': nave['nombre'], 
                                   'Prioridad': nave['prioridad'], 'Cliente': cl, 'Producto': bp,
                                   'Cuadrillas': vy/SCALE, 'Producción': (vy/SCALE)*BASE_RATES[bp]['admin']})
                    # Turnos
                    for s in range(3):
                        vx = solver.Value(x[d, s, n_idx, p_key])
                        if vx > 0:
                            res.append({'Día': DAYS[d], 'Turno': SHIFTS_ORDER[s+1], 'Nave': nave['nombre'], 
                                       'Prioridad': nave['prioridad'], 'Cliente': cl, 'Producto': bp,
                                       'Cuadrillas': vx/SCALE, 'Producción': (vx/SCALE)*BASE_RATES[bp]['turno']})
        return pd.DataFrame(res), status
    return None, status

# --- 5. VISUALIZACIÓN ---
st.title("PLANIFICACIÓN OPERATIVA")
st.caption("Regla aplicada: Una cuadrilla no mezcla productos distintos, pero puede mezclar clientes.")

if run_opt:
    with st.spinner("Optimizando..."):
        df, stt = solve_multinave(st.session_state.naves_list)
    
    if df is not None:
        def fmt(r):
            pc = {1:'#4caf50', 5:'#d32f2f'}.get(r['Prioridad'], '#ffeb3b')
            cc = "#0d47a1" if r['Cliente'] == "ARAUCO" else "#1b5e20"
            crews = f"{r['Cuadrillas']:.1f}".rstrip('0').rstrip('.')
            return (f"<div class='cell-box'><div class='cell-header'>"
                    f"<span style='background:{pc}; color:white; border-radius:50%; width:16px; height:16px; display:flex; justify-content:center; align-items:center; font-size:0.7em;'>{r['Prioridad']}</span>"
                    f"<span style='background:{cc}; color:white; border-radius:3px; padding:0 3px; font-size:0.7em;'>{r['Cliente'][:3]}</span></div>"
                    f"<div class='cell-body'>👥{crews} 📦{int(r['Producción'])}</div></div>")

        df['Info'] = df.apply(fmt, axis=1)
        piv = df.pivot_table(index=['Nave', 'Producto'], columns=['Día', 'Turno'], values='Info', aggfunc=''.join)
        piv = piv.reindex(columns=pd.MultiIndex.from_product([DAYS, SHIFTS_ORDER])).fillna("-")
        
        st.markdown("""<style>
            .cell-box {display: flex; flex-direction: column; align-items: center; gap: 2px;}
            .cell-header {display: flex; gap: 3px;}
            .cell-body {background: #e3f2fd; padding: 2px 4px; border-radius: 4px; font-size: 0.85em;}
            table {width: 100%; border-collapse: collapse;}
            th, td {border: 1px solid #ddd; padding: 4px; text-align: center;}
            thead tr:first-child th {background: #263238; color: white;}
        </style>""", unsafe_allow_html=True)
        
        st.write(piv.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.error("No hay solución factible.")
