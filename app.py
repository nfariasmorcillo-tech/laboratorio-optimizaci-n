import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np

# Configuración de la página web
st.set_page_config(page_title="Laboratorio de Optimización Avanzada", page_icon="🧮", layout="wide")

st.title("🧮 Laboratorio de Optimización con Restricciones Mixtas")
st.markdown("Análisis avanzado de condiciones de **Kuhn-Tucker** y **Hessiano Orlado** basado en la teoría de restricciones mixtas.")

# =========================================================================
# BARRA LATERAL: INGRESO FLEXIBLE DE RESTRICCIONES MULTIPLES
# =========================================================================
st.sidebar.header("📥 Configuración del Problema")

funcion_str = st.sidebar.text_input("Función Objetivo f(x,y):", value="x - y**2")
variables_str = st.sidebar.text_input("Variables libres (separadas por coma):", value="x, y")

st.sidebar.subheader("🔗 Restricciones de Igualdad")
st.sidebar.markdown("Expresadas como $h(x,y) = b \implies h(x,y) - b = 0$")
igualdades_str = st.sidebar.text_input("Restricciones de Igualdad (separadas por ';'):", value="x**2 + y**2 - 10")

st.sidebar.subheader("📐 Restricciones de Desigualdad")
st.sidebar.markdown("Expresadas de forma estándar como $g(x,y) \le c \implies g(x,y) - c \le 0$")
desigualdades_str = st.sidebar.text_input("Restricciones de Desigualdad (separadas por ';'):", value="-x; 1 - y")

st.sidebar.info("**Nota de sintaxis:**\n* Separa múltiples restricciones usando punto y coma (`;`).\n* Ejemplo: `-x` significa $-x \le 0 \implies x \ge 0$.")

calcular = st.sidebar.button("Calcular Óptimos con Kuhn-Tucker", type="primary")

# =========================================================================
# PROCESAMIENTO MATEMÁTICO
# =========================================================================
if calcular:
    try:
        # 1. Parsear Variables
        vars_list = [sp.Symbol(v.strip()) for v in variables_str.split(',') if v.strip()]
        if len(vars_list) != 2:
            st.error("Por favor, introduce exactamente 2 variables para el análisis geométrico plano.")
        else:
            x, y = vars_list[0], vars_list[1]
            f = sp.sympify(funcion_str)

            # Parsear listas de restricciones
            h_list = [sp.sympify(h.strip()) for h in igualdades_str.split(';') if h.strip()]
            g_list = [sp.sympify(g.strip()) for g in desigualdades_str.split(';') if g.strip()]

            # Presentación de ecuaciones base
            st.header("1. Estructura del Problema de Optimización")
            col_f, col_r = st.columns(2)
            with col_f:
                st.latex(rf"\max / \min \quad f({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(f)}")
            with col_r:
                st.markdown("**Restricciones activadas:**")
                for idx, h in enumerate(h_list):
                    st.latex(rf"h_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(h)} = 0")
                for idx, g in enumerate(g_list):
                    st.latex(rf"g_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(g)} \le 0")

            st.divider()

            # 2. Construcción de multiplicadores y Lagrangiano Generalizado
            st.header("2. Construcción del Lagrangiano Generalizado")
            
            # Crear símbolos dinámicos para multiplicadores
            l_simbolos = [sp.Symbol(f"\\lambda_{i+1}") for i in range(len(h_list))]
            m_simbolos = [sp.Symbol(f"\\mu_{j+1}") for j in range(len(g_list))]

            L = f
            for h, lam in zip(h_list, l_simbolos):
                L -= lam * h
            for g, mu in zip(g_list, m_simbolos):
                L -= mu * g

            st.latex(rf"\mathcal{{L}}({sp.latex(x)}, {sp.latex(y)}, \vec{{\lambda}}, \vec{{\mu}}) = {sp.latex(L)}")

            # 3. Condiciones de Primer Orden (Kuhn-Tucker)
            st.header("3. Condiciones de Kuhn-Tucker (Primer Orden)")
            
            Lx = sp.diff(L, x)
            Ly = sp.diff(L, y)
            
            st.markdown("**Gradiente del Lagrangiano respecto a variables espaciales:**")
            st.latex(rf"\mathcal{{L}}_x = {sp.latex(Lx)} = 0")
            st.latex(rf"\mathcal{{L}}_y = {sp.latex(Ly)} = 0")

            st.markdown("**Condiciones de Holgura Complementaria y signo para desigualdades:**")
            ecuaciones_holgura = []
            for g, mu in zip(g_list, m_simbolos):
                st.latex(rf"{sp.latex(mu)} \cdot ({sp.latex(g)}) = 0, \quad {sp.latex(mu)} \ge 0")
                ecuaciones_holgura.append(mu * g)

            # Resolver el sistema por casos (Restricciones Activas vs Inactivas)
            # Para fines didácticos estables en SymPy, evaluamos las combinaciones de activación de las desigualdades
            puntos_validos = []
            
            # Generar todas las combinaciones binarias posibles de restricciones de desigualdad activas (mu >= 0, g = 0) e inactivas (mu = 0, g < 0)
            num_g = len(g_list)
            for combinacion in range(2**num_g):
                eqs_sistema = [Lx, Ly] + h_list
                sustituciones_caso = {}
                
                for j in range(num_g):
                    if (combinacion >> j) & 1:
                        # Restricción j Activa: g_j = 0, mu_j es libre (se calcula)
                        eqs_sistema.append(g_list[j])
                    else:
                        # Restricción j Inactiva: mu_j = 0
                        sustituciones_caso[m_simbolos[j]] = 0
                
                # Aplicar las condiciones del caso al sistema de gradientes
                eqs_resueltas = [eq.subs(sustituciones_caso) for eq in eqs_sistema]
                variables_a_resolver = [x, y] + l_simbolos + [m_simbolos[j] for j in range(num_g) if (combinacion >> j) & 1]
                
                try:
                    sols = sp.solve(eqs_resueltas, variables_a_resolver, dict=True)
                    for sol in sols:
                        # Rellenar los valores fijos del caso
                        for k, v in sustituciones_caso.items():
                            sol[k] = sp.simplify(v)
                        # Verificar si la solución hallada es real y cumple las inecuaciones originales del PDF
                        if sol[x].is_real and sol[y].is_real:
                            cumple_restricciones = True
                            
                            # Verificar que las desigualdades se cumplan: g(x,y) <= 0
                            for g_expr in g_list:
                                if float(g_expr.subs(sol).evalf()) > 1e-5:
                                    cumple_restricciones = False
                            
                            # Verificar signo del multiplicador de Kuhn-Tucker: mu >= 0
                            for mu_sym in m_simbolos:
                                if float(sol[mu_sym].evalf()) < -1e-5:
                                    cumple_restricciones = False
                                    
                            if cumple_restricciones and sol not in puntos_validos:
                                puntos_validos.append(sol)
                except Exception:
                    pass

            # 4. Mostrar Resultados Analíticos
            st.header("4. Puntos Críticos y Clasificación de Fronteras")
            if len(puntos_validos) == 0:
                st.warning("No se encontraron puntos críticos reales que satisfagan simultáneamente las condiciones de Kuhn-Tucker.")
            else:
                datos_tabla = []
                for idx, p in enumerate(puntos_validos):
                    val_f = float(f.subs(p).evalf())
                    
                    # Construir la cadena de visualización de multiplicadores
                    mult_str = ", ".join([f"λ_{i+1}={float(p[l].evalf()):.2f}" for i, l in enumerate(l_simbolos)])
                    mu_str = ", ".join([f"μ_{j+1}={float(p[m].evalf()):.2f}" for j, m in enumerate(m_simbolos)])
                    
                    # Identificar cuáles restricciones están activas (g == 0) para armar el Hessiano Orlado del PDF
                    rest_activas_g = [g for g in g_list if abs(float(g.subs(p).evalf())) < 1e-4]
                    
                    # --- CONSTRUCCIÓN DEL HESSIANO ORLADO SEGÚN EL PDF (PÁGINA 30) ---
                    # El PDF define: H_orlado = [[0, U], [U^T, HL]]
                    # Donde U es el gradiente de todas las restricciones activas (igualdades + desigualdades activas)
                    total_restricciones_activas = h_list + rest_activas_g
                    k_l = len(total_restricciones_activas)
                    
                    # Inicializar Hessiano Orlado
                    if k_l > 0:
                        # Gradientes de restricciones activas respecto a x, y
                        U_list = []
                        for r in total_restricciones_activas:
                            U_list.append([sp.diff(r, x), sp.diff(r, y)])
                        U_mat = sp.Matrix(U_list)
                        
                        # Segundas derivadas del lagrangiano (HL)
                        Lxx = sp.diff(Lx, x)
                        Lyy = sp.diff(Ly, y)
                        Lxy = sp.diff(Lx, y)
                        HL = sp.Matrix([[Lxx, Lxy], [Lxy, Lyy]])
                        
                        # Construcción por bloques de la matriz orlada
                        cero_block = sp.zeros(k_l, k_l)
                        fila_superior = cero_block.row_join(U_mat)
                        fila_inferior = U_mat.T.row_join(HL)
                        H_orlado_completo = fila_superior.col_join(fila_inferior)
                        
                        # Evaluar el determinante en el punto crítico
                        H_evaluado = H_orlado_completo.subs(p)
                        det_hor = float(H_evaluado.det().evalf())
                    else:
                        # Si no hay restricciones activas actúa como optimización libre (Hessiano estándar)
                        Lxx = sp.diff(Lx, x).subs(p)
                        Lyy = sp.diff(Ly, y).subs(p)
                        Lxy = sp.diff(Lx, y).subs(p)
                        H_evaluado = sp.Matrix([[Lxx, Lxy], [Lxy, Lyy]])
                        det_hor = float(H_evaluado.det().evalf())

                    # Criterio de clasificación de menores del Hessiano Orlado (PDF Pág 30)
                    # Para n=2 variables libres:
                    if k_l == 1:
                        # Con 1 restricción activa, el determinante de la matriz orlada 3x3 decide directamente:
                        # Si det > 0 es Máximo, si det < 0 es Mínimo
                        tipo_optimo = "🔵 Máximo Local Condicionado" if det_hor > 0 else "🟢 Mínimo Local Condicionado"
                    elif k_l == 0:
                        # Hessiano común sin restricciones
                        fxx_p = float(H_evaluado[0,0].evalf())
                        tipo_optimo = "🟢 Mínimo Libre" if (det_hor > 0 and fxx_p > 0) else "🔵 Máximo Libre" if (det_hor > 0 and fxx_p < 0) else "🔴 Punto Silla"
                    else:
                        # Si k+l >= n (sistema completamente determinado en frontera/vértice)
                        tipo_optimo = "📐 Vértice / Punto de Frontera Rígido"

                    datos_tabla.append({
                        "ID": f"P{idx+1}",
                        "Coordenada (x, y)": f"({float(sol[x].evalf()):.3f}, {float(sol[y].evalf()):.3f})",
                        "Multiplicadores Igualdad": mult_str if h_list else "Ninguno",
                        "Multiplicadores Desigualdad (KKT)": mu_str if g_list else "Ninguno",
                        "Det |H| Orlado": f"{det_hor:.3f}",
                        "Clasificación Teórica": tipo_optimo,
                        "Valor Objetivo f(x,y)": f"{val_f:.4f}"
                    })

                df_resultados = pd.DataFrame(datos_tabla)
                st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                
                # Desglose algebraico individual por punto encontrado
                st.subheader("🔍 Matrices del Hessiano Orlado Evaluadas de forma Cruzada")
                for idx, p in enumerate(puntos_validos):
                    rest_activas_g = [g for g in g_list if abs(float(g.subs(p).evalf())) < 1e-4]
                    total_restricciones_activas = h_list + rest_activas_g
                    k_l = len(total_restricciones_activas)
                    
                    if k_l > 0:
                        U_list = [[sp.diff(r, x), sp.diff(r, y)] for r in total_restricciones_activas]
                        U_mat = sp.Matrix(U_list)
                        HL = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]])
                        H_orlado_completo = sp.zeros(k_l, k_l).row_join(U_mat).col_join(U_mat.T.row_join(HL))
                        mat_latex = H_orlado_completo.subs(p)
                    else:
                        mat_latex = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]]).subs(p)
                        
                    st.markdown(f"**Matriz Estructural para el Punto P{idx+1}:**")
                    st.latex(rf"\overline{{H}}_{{P_{idx+1}}} = {sp.latex(mat_latex)} \implies |\overline{{H}}| = {sp.latex(mat_latex.det())}")

    except Exception as e:
        st.error(f"Error analítico en la evaluación: {e}. Revisa que las funciones estén bien escritas y separadas por `;`.")
