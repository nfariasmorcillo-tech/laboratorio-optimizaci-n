import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np

# Configuración de la página web
st.set_config = st.set_page_config(page_title="Laboratorio de Optimización Avanzada", page_icon="🧮", layout="wide")

st.title("🧮 Laboratorio de Optimización con Restricciones Mixtas")
st.markdown("Análisis avanzado paso a paso mediante el **Árbol de Supuestos de Kuhn-Tucker**.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Configuración del Problema")

funcion_str = st.sidebar.text_input("Función Objetivo f(x,y):", value="x - y**2")
variables_str = st.sidebar.text_input("Variables libres (separadas por coma):", value="x, y")

st.sidebar.subheader("🔗 Restricciones de Igualdad")
st.sidebar.markdown("Forma estándar: $h(x,y) = 0$")
igualdades_str = st.sidebar.text_input("Restricciones de Igualdad (separadas por ';'):", value="x**2 + y**2 - 10")

st.sidebar.subheader("📐 Restricciones de Desigualdad")
st.sidebar.markdown("Forma estándar: $g(x,y) \le 0$")
desigualdades_str = st.sidebar.text_input("Restricciones de Desigualdad (separadas por ';'):", value="-x; 1 - y")

st.sidebar.info("**Nota pedagógica:** El sistema evaluará ramificando los supuestos de $\mu_i > 0$ (activa) y $\mu_i = 0$ (inactiva), mostrando las contradicciones encontradas.")

calcular = st.sidebar.button("Calcular con Árbol de Supuestos", type="primary")

# =========================================================================
# MOTOR ALGEBRAICO CON REGISTRO DE CONTRADICCIONES
# =========================================================================
if calcular:
    try:
        # Parsear variables y funciones
        vars_list = [sp.Symbol(v.strip()) for v in variables_str.split(',') if v.strip()]
        if len(vars_list) != 2:
            st.error("Por favor, introduce exactamente 2 variables libres (ej: x, y).")
        else:
            x, y = vars_list[0], vars_list[1]
            f = sp.sympify(funcion_str)

            h_list = [sp.sympify(h.strip()) for h in igualdades_str.split(';') if h.strip()]
            g_list = [sp.sympify(g.strip()) for g in desigualdades_str.split(';') if g.strip()]

            st.header("1. Estructura del Problema")
            col_f, col_r = st.columns(2)
            with col_f:
                st.latex(rf"\max / \min \quad f({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(f)}")
            with col_r:
                st.markdown("**Restricciones iniciales:**")
                for idx, h in enumerate(h_list):
                    st.latex(rf"h_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(h)} = 0")
                for idx, g in enumerate(g_list):
                    st.latex(rf"g_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(g)} \le 0")

            st.divider()

            # Símbolos para multiplicadores
            l_simbolos = [sp.Symbol(f"\\lambda_{i+1}") for i in range(len(h_list))]
            m_simbolos = [sp.Symbol(f"\\mu_{j+1}") for j in range(len(g_list))]

            # Construcción del Lagrangiano
            L = f
            for h, lam in zip(h_list, l_simbolos):
                L -= lam * h
            for g, mu in zip(g_list, m_simbolos):
                L -= mu * g

            Lx = sp.diff(L, x)
            Ly = sp.diff(L, y)

            st.header("2. Condiciones de Primer Orden")
            st.latex(rf"\mathcal{{L}}_x = {sp.latex(Lx)} = 0")
            st.latex(rf"\mathcal{{L}}_y = {sp.latex(Ly)} = 0")

            st.header("3. Rastreo del Árbol de Supuestos (Kuhn-Tucker)")
            st.markdown("De acuerdo con la teoría, para cada restricción de desigualdad se asume inicialmente que está activa ($\mu_i > 0$). Si esto genera una contradicción matemática en el sistema de ecuaciones, se concluye que la restricción es inactiva ($\mu_i = 0$).")

            num_g = len(g_list)
            puntos_validos = []
            historial_supuestos = []

            # Evaluar sistemáticamente las combinaciones de supuestos
            # Cada combinación representa un escenario del árbol de decisión
            for combinacion in range(2**num_g):
                supuestos_texto = []
                eqs_sistema = [Lx, Ly] + h_list
                sustituciones_caso = {}
                
                # Configurar el escenario del supuesto
                for j in range(num_g):
                    if (combinacion >> j) & 1:
                        # Supuesto: Restricción Activa (mu_j > 0, g_j = 0)
                        eqs_sistema.append(g_list[j])
                        supuestos_texto.append(rf"\mu_{j+1} > 0 \text{{ (g}}_{j+1}\text{{ activa: }} {sp.latex(g_list[j])} = 0)")
                    else:
                        # Supuesto alterno por descarte: Restricción Inactiva (mu_j = 0)
                        sustituciones_caso[m_simbolos[j]] = 0
                        supuestos_texto.append(rf"\mu_{j+1} = 0 \text{{ (g}}_{j+1}\text{{ inactiva)}}")

                texto_caso_latex = " y ".join(supuestos_texto)
                
                # Aplicar sustituciones de inactivas a las ecuaciones del gradiente
                eqs_resueltas = [eq.subs(sustituciones_caso) for eq in eqs_sistema]
                variables_a_resolver = [x, y] + l_simbolos + [m_simbolos[j] for j in range(num_g) if (combinacion >> j) & 1]

                with st.expander(rf"🌿 Evaluando Supuesto: ${texto_caso_latex}$"):
                    st.markdown("**Sistema de ecuaciones resultante para este camino:**")
                    for eq in eqs_resueltas:
                        st.latex(rf"{sp.latex(eq)} = 0")
                    
                    try:
                        # Intentar resolver el sistema algebraico
                        sols = sp.solve(eqs_resueltas, variables_a_resolver, dict=True)
                        
                        if not sols:
                            st.error("❌ **Contradicción / Inconsistencia:** El sistema no tiene soluciones bajo este supuesto. Por lo tanto, este camino queda descartado.")
                            historial_supuestos.append({
                                "Supuesto Evaluado": f"${texto_caso_latex}$",
                                "Estado": "❌ Contradicción Algebraica",
                                "Detalle / Acción": "Sistema inconsistente. Forzar multiplicadores activos a cero."
                            })
                            continue

                        # Si hay soluciones, validar restricciones lógicas colaterales (KKT)
                        caminos_validos_en_este_supuesto = 0
                        for sol in sols:
                            # Completar los valores de los multiplicadores asumidos como 0
                            for k, v in sustituciones_caso.items():
                                sol[k] = sp.simplify(v)
                            
                            if sol[x].is_real and sol[y].is_real:
                                cumple_kkt = True
                                motivo_descarte = ""

                                # Validación 1: Verificar que no viole las demás desigualdades g(x,y) <= 0
                                for idx_g, g_expr in enumerate(g_list):
                                    val_g = float(g_expr.subs(sol).evalf())
                                    if val_g > 1e-4:
                                        cumple_kkt = False
                                        motivo_descarte = f"Viola la restricción g_{idx_g+1} ({val_g:.2f} > 0)"

                                # Validación 2: Verificar la condición de signo mu_i >= 0
                                for mu_sym in m_simbolos:
                                    val_mu = float(sol[mu_sym].evalf())
                                    if val_mu < -1e-4:
                                        cumple_kkt = False
                                        motivo_descarte = f"Multiplicador negativo ({sp.latex(mu_sym)} = {val_mu:.2f} < 0)"

                                if cumple_kkt:
                                    sol_copia = sol.copy()
                                    if sol_copia not in puntos_validos:
                                        puntos_validos.append(sol_copia)
                                    caminos_validos_en_este_supuesto += 1
                                    st.success(rf"✅ **Solución Factible Hallada:** $({float(sol[x].evalf()):.3f}, {float(sol[y].evalf()):.3f})$ con $\mu = {[float(sol[m].evalf()) for m in m_simbolos]}$")
                                else:
                                    st.warning(rf"⚠️ **Descartado por criterio KKT:** Solución matemática hallada pero inválida. Motivo: {motivo_descarte}.")
                        
                        if caminos_validos_en_este_supuesto > 0:
                            historial_supuestos.append({
                                "Supuesto Evaluado": f"${texto_caso_latex}$",
                                "Estado": "✅ Factible",
                                "Detalle / Acción": f"Aporta {caminos_validos_en_este_supuesto} punto(s) crítico(s) válido(s)."
                            })
                        else:
                            historial_supuestos.append({
                                "Supuesto Evaluado": f"${texto_caso_latex}$",
                                "Estado": "⚠️ Descartado",
                                "Detalle / Acción": "Raíces fuera de la región factible o signos de multiplicadores incorrectos."
                            })

                    except Exception as e:
                        st.error(f"Error analítico al resolver este camino: {e}")

            # Mostrar la tabla resumen del árbol de decisiones pedagógico
            st.subheader("📋 Tabla Resumen del Proceso Analítico (Árbol de Supuestos)")
            df_arbol = pd.DataFrame(historial_supuestos)
            st.dataframe(df_arbol, use_container_width=True, hide_index=True)

            st.divider()

            # =========================================================================
            # RESULTADOS FINALES Y HESSIANO ORLADO
            # =========================================================================
            st.header("4. Matriz de Resultados Finales y Segundas Derivadas")
            
            if len(puntos_validos) == 0:
                st.warning("No quedan puntos críticos sobrevivientes que cumplan las condiciones del espacio factible.")
            else:
                datos_tabla_final = []
                for idx, p in enumerate(puntos_validos):
                    val_f = float(f.subs(p).evalf())
                    
                    mult_str = ", ".join([f"λ_{i+1}={float(p[l].evalf()):.2f}" for i, l in enumerate(l_simbolos)])
                    mu_str = ", ".join([f"μ_{j+1}={float(p[m].evalf()):.2f}" for j, m in enumerate(m_simbolos)])
                    
                    # Detectar restricciones activas reales para armar el Hessiano Orlado exacto
                    rest_activas_g = [g for g in g_list if abs(float(g.subs(p).evalf())) < 1e-4]
                    total_activas = h_list + rest_activas_g
                    k_l = len(total_activas)
                    
                    if k_l > 0:
                        U_list = [[sp.diff(r, x), sp.diff(r, y)] for r in total_activas]
                        U_mat = sp.Matrix(U_list)
                        HL = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]])
                        H_orlado = sp.zeros(k_l, k_l).row_join(U_mat).col_join(U_mat.T.row_join(HL))
                        det_hor = float(H_orlado.subs(p).det().evalf())
                        
                        tipo_optimo = "🔵 Máximo Condicionado" if det_hor > 0 else "🟢 Mínimo Condicionado"
                    else:
                        HL = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]])
                        det_hor = float(HL.subs(p).det().evalf())
                        fxx_p = float(HL.subs(p)[0,0].evalf())
                        tipo_optimo = "🟢 Mínimo Libre" if (det_hor > 0 and fxx_p > 0) else "🔵 Máximo Libre" if (det_hor > 0 and fxx_p < 0) else "🔴 Punto Silla"

                    datos_tabla_final.append({
                        "ID": f"P{idx+1}",
                        "Coordenada (x, y)": f"({float(p[x].evalf()):.3f}, {float(p[y].evalf()):.3f})",
                        "Igualdades (λ)": mult_str if h_list else "---",
                        "Desigualdades (μ)": mu_str if g_list else "---",
                        "Det |H| Orlado": f"{det_hor:.3f}",
                        "Clasificación": tipo_optimo,
                        "Valor f(x,y)": f"{val_f:.4f}"
                    })

                df_final = pd.DataFrame(datos_tabla_final)
                st.dataframe(df_final, use_container_width=True, hide_index=True)

                # Mostrar las matrices del Hessiano Orlado correspondientes
                st.subheader("🔍 Estructura del Hessiano Orlado Evaluado por Punto")
                for idx, p in enumerate(puntos_validos):
                    rest_activas_g = [g for g in g_list if abs(float(g.subs(p).evalf())) < 1e-4]
                    total_activas = h_list + rest_activas_g
                    k_l = len(total_activas)
                    
                    if k_l > 0:
                        U_list = [[sp.diff(r, x), sp.diff(r, y)] for r in total_activas]
                        U_mat = sp.Matrix(U_list)
                        HL = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]])
                        H_orlado = sp.zeros(k_l, k_l).row_join(U_mat).col_join(U_mat.T.row_join(HL))
                        mat_print = H_orlado.subs(p)
                    else:
                        mat_print = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]]).subs(p)
                    
                    st.markdown(f"**Matriz del punto P{idx+1}:**")
                    st.latex(rf"\overline{{H}}_{{P_{idx+1}}} = {sp.latex(mat_print)} \implies |\overline{{H}}| = {float(mat_print.det().evalf()):.3f}")

    except Exception as e:
        st.error(f"Error general en el procesamiento analítico: {e}. Asegúrate de separar las restricciones con punto y coma (`;`).")
