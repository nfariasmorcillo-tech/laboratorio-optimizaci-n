import streamlit as st
import sympy as sp
import pandas as pd

# Configuración de la página web
st.set_config = st.set_page_config(page_title="Laboratorio de Optimización Avanzada", page_icon="🧮", layout="wide")

st.title("🧮 Laboratorio de Optimización con Restricciones Mixtas")
st.markdown("Análisis avanzado paso a paso mediante el **Árbol de Supuestos de Kuhn-Tucker**.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Configuración del Problema")

funcion_str = st.sidebar.text_input("Función Objetivo f(x,y):", value="4*x - 2*x**2 - 2*y**2")
variables_str = st.sidebar.text_input("Variables libres (separadas por coma):", value="x, y")

st.sidebar.subheader("🔗 Restricciones de Igualdad")
st.sidebar.markdown("Forma estándar: $h(x,y) = 0$")
igualdades_str = st.sidebar.text_input("Restricciones de Igualdad (separadas por ';'):", value="")

st.sidebar.subheader("📐 Restricciones de Desigualdad")
st.sidebar.markdown("Forma estándar: $g(x,y) \le 0$")
desigualdades_str = st.sidebar.text_input("Restricciones de Desigualdad (separadas por ';'):", value="x**2 + y**2 - 25")

st.sidebar.info("**Nota pedagógica:** Lagrangiano corregido en forma aditiva: $\mathcal{L} = f + \lambda(h) + \mu(g)$.")

calcular = st.sidebar.button("Calcular con Desglose Analítico", type="primary")

# =========================================================================
# MOTOR ALGEBRAICO CON LAGRANGIANO CORREGIDO Y ANÁLISIS DE FACTORES
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
                st.markdown("**Restricciones iniciales ($h=0$, $g \le 0$):**")
                for idx, h in enumerate(h_list):
                    st.latex(rf"h_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(h)} = 0")
                for idx, g in enumerate(g_list):
                    st.latex(rf"g_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(g)} \le 0")

            st.divider()

            # Símbolos para multiplicadores
            l_simbolos = [sp.Symbol(f"\\lambda_{i+1}") for i in range(len(h_list))]
            m_simbolos = [sp.Symbol(f"\\mu_{j+1}") for j in range(len(g_list))]

            # FORMULA CORREGIDA: Construcción del Lagrangiano f + lambda*(h) + mu*(g)
            L = f
            for h, lam in zip(h_list, l_simbolos):
                L += lam * h
            for g, mu in zip(g_list, m_simbolos):
                L += mu * g

            Lx = sp.diff(L, x)
            Ly = sp.diff(L, y)

            st.header("2. Condiciones de Primer Orden")
            st.markdown("**Expresión del Lagrangiano construido:**")
            st.latex(rf"\mathcal{{L}} = {sp.latex(L)}")
            
            st.markdown("**Gradientes del sistema:**")
            st.latex(rf"\mathcal{{L}}_x = {sp.latex(Lx)} = 0")
            st.latex(rf"\mathcal{{L}}_y = {sp.latex(Ly)} = 0")

            st.header("3. Rastreo del Árbol de Supuestos (Kuhn-Tucker)")
            
            num_g = len(g_list)
            puntos_validos = []
            historial_supuestos = []

            # Evaluar combinaciones de restricciones activas/inactivas
            iteraciones = 2**num_g if num_g > 0 else 1
            for combinacion in range(iteraciones):
                supuestos_texto = []
                eqs_base = [Lx, Ly] + h_list
                sustituciones_caso = {}
                
                for j in range(num_g):
                    if (combinacion >> j) & 1:
                        eqs_base.append(g_list[j])
                        supuestos_texto.append(rf"\mu_{j+1} > 0")
                    else:
                        sustituciones_caso[m_simbolos[j]] = 0
                        supuestos_texto.append(rf"\mu_{j+1} = 0")

                texto_caso_latex = " \quad y \quad ".join(supuestos_texto) if supuestos_texto else "Optimización Estándar"
                eqs_con_reemplazo = [eq.subs(sustituciones_caso) for eq in eqs_base]
                variables_a_resolver = [x, y] + l_simbolos + [m_simbolos[j] for j in range(num_g) if (combinacion >> j) & 1]

                with st.expander(rf"🌿 Evaluando Ramificación: ${texto_caso_latex}$"):
                    st.markdown("**Sistema algebraico con multiplicadores fijos sustituidos:**")
                    for eq in eqs_con_reemplazo:
                        st.latex(rf"{sp.latex(eq)} = 0")
                    
                    # --- DETECTOR Y DESCOMPOSITOR DE FACTORES EXCLUYENTES ---
                    sistemas_a_evaluar = []
                    factorizacion_ocurrida = False

                    for idx_eq, eq in enumerate(eqs_con_reemplazo):
                        eq_factored = sp.factor(eq)
                        if isinstance(eq_factored, sp.Mul):
                            factores = [arg for arg in eq_factored.args if not arg.is_number]
                            if len(factores) > 1:
                                factorizacion_ocurrida = True
                                st.markdown(rf"🔍 **Factorización Crítica Detectada:** Ecuación {idx_eq+1} se descompone en ${sp.latex(eq_factored)} = 0$")
                                
                                for i_f, factor_activo in enumerate(factores):
                                    nuevas_eqs = []
                                    for k_eq, e_comp in enumerate(eqs_con_reemplazo):
                                        if k_eq == idx_eq:
                                            nuevas_eqs.append(factor_activo)
                                        else:
                                            nuevas_eqs.append(e_comp)
                                    
                                    otros_factores = [factores[m] for m in range(len(factores)) if m != i_f]
                                    condicion_excluyente = " y ".join([f"${sp.latex(of)} \\neq 0$" for of in otros_factores])
                                    sistemas_a_evaluar.append((nuevas_eqs, f"Sub-caso: ${sp.latex(factor_activo)} = 0$ (asumiendo {condicion_excluyente})"))
                                break

                    if not factorizacion_ocurrida:
                        sistemas_a_evaluar.append((eqs_con_reemplazo, "Análisis directo de bloque simultáneo"))

                    caminos_validos_en_este_supuesto = 0
                    
                    for eqs_sub_caso, desc_sub_caso in sistemas_a_evaluar:
                        st.markdown(f"**👉 Evaluando {desc_sub_caso}:**")
                        if factorizacion_ocurrida:
                            st.markdown("*Ecuaciones reestructuradas de la ramificación:*")
                            for eq_s in eqs_sub_caso:
                                st.latex(rf"{sp.latex(eq_s)} = 0")

                        try:
                            sols = sp.solve(eqs_sub_caso, variables_a_resolver, dict=True)
                            
                            if not sols:
                                st.caption("❌ *Resultado de sub-caso:* Contradicción analítica directa (Incompatible).")
                                continue
                            
                            for sol in sols:
                                for k, v in sustituciones_caso.items():
                                    sol[k] = sp.simplify(v)
                                
                                if sol[x].is_real and sol[y].is_real:
                                    cumple_kkt = True
                                    motivos_descarte = []

                                    st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;• Coordenada matemática hallada: $x = {sp.latex(sol[x])}$, $y = {sp.latex(sol[y])}$")

                                    # Validar región factible g(x,y) <= 0
                                    for idx_g, g_expr in enumerate(g_list):
                                        val_g = float(g_expr.subs(sol).evalf())
                                        if val_g > 1e-4:
                                            cumple_kkt = False
                                            motivos_descarte.append(f"g_{idx_g+1} > 0 ({val_g:.2f})")

                                    # Validar signos de Kuhn-Tucker mu >= 0
                                    for mu_sym in m_simbolos:
                                        if mu_sym in sol:
                                            val_mu = float(sol[mu_sym].evalf())
                                            if val_mu < -1e-4:
                                                cumple_kkt = False
                                                motivos_descarte.append(rf"{sp.latex(mu_sym)} < 0 ({val_mu:.2f})")

                                    if cumple_kkt:
                                        sol_copia = sol.copy()
                                        if sol_copia not in puntos_validos:
                                            puntos_validos.append(sol_copia)
                                        caminos_validos_en_este_supuesto += 1
                                        st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎯 **Estatus:** ¡Punto crítico válido aprobado!")
                                    else:
                                        st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⚠️ **Estatus:** Raíz descartada por: {', '.join(motivos_descarte)}.")

                        except Exception as e:
                            st.error(f"Error procesando la sub-rama analítica: {e}")

                    if caminos_validos_en_este_supuesto > 0:
                        historial_supuestos.append({
                            "Supuesto Evaluado": f"${texto_caso_latex}$",
                            "Fase Algebraica": "✅ Consistente (Desglosado)",
                            "Resultado Final": f"Aporta {caminos_validos_en_este_supuesto} punto(s)"
                        })
                    else:
                        historial_supuestos.append({
                            "Supuesto Evaluado": f"${texto_caso_latex}$",
                            "Fase Algebraica": "❌ Contradicción / Invalidez",
                            "Resultado Final": "Descartado"
                        })

            # Tabla resumen del árbol
            st.subheader("📋 Tabla Resumen del Proceso Analítico")
            df_arbol = pd.DataFrame(historial_supuestos)
            st.dataframe(df_arbol, use_container_width=True, hide_index=True)

            st.divider()

            # =========================================================================
            # RESULTADOS FINALES Y SEGUNDAS DERIVADAS (HESSIANO ORLADO)
            # =========================================================================
            st.header("4. Matriz de Resultados Finales y Segundas Derivadas")
            
            if len(puntos_validos) == 0:
                st.warning("No se encontraron puntos críticos que satisfagan las restricciones impuestas.")
            else:
                datos_tabla_final = []
                for idx, p in enumerate(puntos_validos):
                    val_f = float(f.subs(p).evalf())
                    
                    mult_str = ", ".join([f"λ_{i+1}={float(p[l].evalf()):.2f}" for i, l in enumerate(l_simbolos)])
                    mu_str = ", ".join([f"μ_{j+1}={float(p[m].evalf()):.2f}" for j, m in enumerate(m_simbolos)])
                    
                    # Detectar restricciones que se activan numéricamente en la coordenada (g == 0)
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

                # Matrices detalladas en formato matemático LaTeX
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
        st.error(f"Error general de ejecución: {e}. Por favor valida las expresiones matemáticas ingresadas.")
