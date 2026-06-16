import streamlit as st
import sympy as sp
import pandas as pd

# Configuración de la página web
st.set_page_config(page_title="Laboratorio de Optimización Avanzada", page_icon="🧮", layout="wide")

st.title("🧮 Laboratorio de Optimización con Restricciones Mixtas")
st.markdown("Análisis avanzado paso a paso mediante el **Árbol de Supuestos de Kuhn-Tucker (KKT)**.")

# =========================================================================
# BARRA LATERAL: INGRESO DE DATOS
# =========================================================================
st.sidebar.header("📥 Configuración del Problema")

objetivo = st.sidebar.selectbox("Objetivo de Optimización:", ["Maximizar", "Minimizar"])

funcion_str = st.sidebar.text_input("Función Objetivo f(x,y):", value="x**2 - 2*x*y + 2*y")
variables_str = st.sidebar.text_input("Variables libres (separadas por coma):", value="x, y")

st.sidebar.subheader("🔗 Restricciones de Igualdad")
st.sidebar.markdown("Forma estándar: $h(x,y) = 0$")
igualdades_str = st.sidebar.text_input("Restricciones de Igualdad (separadas por ';'):", value="")

st.sidebar.subheader("📐 Restricciones de Desigualdad")
st.sidebar.markdown("Forma estándar: $g(x,y) \le 0$")
desigualdades_str = st.sidebar.text_input("Restricciones de Desigualdad (separadas por ';'):", value="-x; x-3; -y; y-2")

st.sidebar.info(f"**Criterio KKT Activo ({objetivo}):**\n"
                f"Lagrangiano aditivo: $\mathcal{{L}} = f + \lambda(h) + \mu(g)$.\n"
                f"• Si se Maximiza, se exige $\mu \le 0$.\n"
                f"• Si se Minimiza, se exige $\mu \ge 0$.")

calcular = st.sidebar.button("Calcular Optimización Condicionada", type="primary")

# =========================================================================
# MOTOR ALGEBRAICO CON EXPLICACIÓN PASO A PASO DE RESOLUCIÓN
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

            st.header(f"1. Estructura del Problema ({objetivo})")
            col_f, col_r = st.columns(2)
            with col_f:
                st.latex(rf"\text{{{objetivo}}}\quad f({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(f)}")
            with col_r:
                st.markdown("**Restricciones en formato estándar:**")
                for idx, h in enumerate(h_list):
                    st.latex(rf"h_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(h)} = 0")
                for idx, g in enumerate(g_list):
                    st.latex(rf"g_{idx+1}({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(g)} \le 0")

            st.divider()

            # Símbolos para multiplicadores
            l_simbolos = [sp.Symbol(f"\\lambda_{i+1}") for i in range(len(h_list))]
            m_simbolos = [sp.Symbol(f"\\mu_{j+1}") for j in range(len(g_list))]

            # Construcción del Lagrangiano Aditivo Universal
            L = f
            for h, lam in zip(h_list, l_simbolos):
                L += lam * h
            for g, mu in zip(g_list, m_simbolos):
                L += mu * g

            Lx = sp.diff(L, x)
            Ly = sp.diff(L, y)

            st.header("2. Condiciones de Primer Orden (CPO)")
            st.latex(rf"\mathcal{{L}} = {sp.latex(L)}")
            st.latex(rf"\mathcal{{L}}_x = {sp.latex(Lx)} = 0")
            st.latex(rf"\mathcal{{L}}_y = {sp.latex(Ly)} = 0")

            st.header("3. Desglose del Árbol de Supuestos Analíticos con Resolución de Sistemas")
            
            num_g = len(g_list)
            puntos_validos = []
            historial_supuestos = []

            iteraciones = 2**num_g if num_g > 0 else 1
            for combinacion in range(iteraciones):
                supuestos_texto = []
                eqs_base = [Lx, Ly] + h_list
                sustituciones_caso = {}
                
                for j in range(num_g):
                    if (combinacion >> j) & 1:
                        eqs_base.append(g_list[j])
                        supuestos_texto.append(rf"\mu_{j+1} \neq 0")
                    else:
                        sustituciones_caso[m_simbolos[j]] = 0
                        supuestos_texto.append(rf"\mu_{j+1} = 0")

                texto_caso_latex = " \quad y \quad ".join(supuestos_texto) if supuestos_texto else "Sin restricciones de desigualdad"
                eqs_con_reemplazo = [eq.subs(sustituciones_caso) for eq in eqs_base]
                variables_a_resolver = [x, y] + l_simbolos + [m_simbolos[j] for j in range(num_g) if (combinacion >> j) & 1]

                with st.expander(rf"🌿 Evaluando Ramificación: ${texto_caso_latex}$"):
                    st.markdown("#### **Sistema Inicial de Ecuaciones Numeradas:**")
                    for idx_eq, eq in enumerate(eqs_con_reemplazo):
                        st.latex(rf"({idx_eq+1})\quad {sp.latex(eq)} = 0")
                    
                    # --- MOTOR DE EXPLICACIÓN Y FACTORIZACIÓN ---
                    sistemas_a_evaluar = []
                    factorizacion_ocurrida = False

                    for idx_eq, eq in enumerate(eqs_con_reemplazo):
                        eq_factored = sp.factor(eq)
                        if isinstance(eq_factored, sp.Mul):
                            factores = [arg for arg in eq_factored.args if not arg.is_number]
                            if len(factores) > 1:
                                factorizacion_ocurrida = True
                                st.markdown(f"**🔍 Detección Algebraica:** La ecuación ({idx_eq+1}) es factorizable como:")
                                st.latex(rf"{sp.latex(eq_factored)} = 0")
                                
                                for i_f, factor_activo in enumerate(factores):
                                    nuevas_eqs = []
                                    for k_eq, e_comp in enumerate(eqs_con_reemplazo):
                                        if k_eq == idx_eq:
                                            nuevas_eqs.append(factor_activo)
                                        else:
                                            nuevas_eqs.append(e_comp)
                                    
                                    otros_factores = [factores[m] for m in range(len(factores)) if m != i_f]
                                    condicion_excluyente = " y ".join([f"${sp.latex(of)} \\neq 0$" for of in otros_factores])
                                    sistemas_a_evaluar.append((nuevas_eqs, f"Sub-caso derivado de ({idx_eq+1}): Asumiendo ${sp.latex(factor_activo)} = 0$ (mientras {condicion_excluyente})", idx_eq+1, factor_activo))
                                break

                    if not factorizacion_ocurrida:
                        sistemas_a_evaluar.append((eqs_con_reemplazo, "Resolución directa del bloque simultáneo lineal/estructural", None, None))

                    caminos_validos_en_este_supuesto = 0
                    
                    for eqs_sub_caso, desc_sub_caso, eq_origen_num, factor_usado in sistemas_a_evaluar:
                        st.markdown(f"##### **👉 {desc_sub_caso}:**")
                        
                        # Explicación paso a paso del desarrollo manual simulado
                        if eq_origen_num is not None:
                            st.markdown(f"1. De la ecuación de origen, evaluamos el factor nulo: ${sp.latex(factor_usado)} = 0$.")
                            st.markdown(f"2. Sustituimos esta equivalencia matemática en el resto de las ecuaciones del sistema:")
                            for idx_s, eq_s in enumerate(eqs_sub_caso):
                                if idx_s + 1 != eq_origen_num:
                                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• Reemplazando en ({idx_s+1}) se reduce a: ${sp.latex(eq_s)} = 0$")
                        else:
                            st.markdown("1. Despejamos el sistema combinando directamente los componentes lineales o determinantes estructurales:")

                        try:
                            sols = sp.solve(eqs_sub_caso, variables_a_resolver, dict=True)
                            
                            if not sols:
                                st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;❌ **Resultado del análisis:** El sistema no tiene soluciones reales consistentes (Contradicción algebraica).")
                                continue
                            
                            for sol in sols:
                                for k, v in sustituciones_caso.items():
                                    sol[k] = sp.simplify(v)
                                
                                if sol[x].is_real and sol[y].is_real:
                                    cumple_kkt = True
                                    motivos_descarte = []

                                    # Mostrar los despejes numéricos finales deducidos
                                    st.markdown(f"3. **Valores calculados para este sub-caso:**")
                                    st.latex(rf"x = {sp.latex(sol[x])}, \quad y = {sp.latex(sol[y])}")
                                    
                                    mult_activos_print = [rf"{sp.latex(m_sym)} = {sp.latex(sol[m_sym])}" for m_sym in variables_a_resolver if m_sym in m_simbolos]
                                    if mult_activos_print:
                                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;• Multiplicadores deducidos: {', '.join([f'${m}$' for m in mult_activos_print])}")

                                    # 1. Validar región factible g(x,y) <= 0
                                    for idx_g, g_expr in enumerate(g_list):
                                        val_g = float(g_expr.subs(sol).evalf())
                                        if val_g > 1e-4:
                                            cumple_kkt = False
                                            motivos_descarte.append(f"g_{idx_g+1} > 0 ({val_g:.2f})")

                                    # 2. Validación de signos según el objetivo seleccionado
                                    for mu_sym in m_simbolos:
                                        if mu_sym in sol:
                                            val_mu = float(sol[mu_sym].evalf())
                                            
                                            if objetivo == "Maximizar" and val_mu > 1e-4:
                                                cumple_kkt = False
                                                motivos_descarte.append(rf"\text{{Multiplicador }} {sp.latex(mu_sym)} = {val_mu:.2f} > 0 \text{{ (Incompatible con Maximización en Lagrangiano aditivo)}}")
                                            
                                            elif objetivo == "Minimizar" and val_mu < -1e-4:
                                                cumple_kkt = False
                                                motivos_descarte.append(rf"\text{{Multiplicador }} {sp.latex(mu_sym)} = {val_mu:.2f} < 0 \text{{ (Incompatible con Minimización en Lagrangiano aditivo)}}")

                                    if cumple_kkt:
                                        sol_copia = sol.copy()
                                        if sol_copia not in puntos_validos:
                                            puntos_validos.append(sol_copia)
                                        caminos_validos_en_este_supuesto += 1
                                        st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🎯 **Estatus:** ¡Punto crítico válido aprobado bajo el criterio de {objetivo}!")
                                    else:
                                        st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;⚠️ **Estatus:** Punto descartado analíticamente debido a: {', '.join(motivos_descarte)}.")

                        except Exception as e:
                            st.error(f"Error procesando la sub-rama: {e}")

                    if caminos_validos_en_este_supuesto > 0:
                        historial_supuestos.append({
                            "Supuesto Evaluado": f"${texto_caso_latex}$",
                            "Fase Algebraica": "✅ Consistente",
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
            # RESULTADOS FINALES Y CLASIFICACIÓN SEGUNDO ORDEN
            # =========================================================================
            st.header("4. Matriz de Resultados Finales Aprobados")
            
            if len(puntos_validos) == 0:
                st.warning(f"No se encontraron puntos críticos óptimos locales que cumplan las condiciones estrictas de {objetivo} para KKT.")
            else:
                datos_tabla_final = []
                for idx, p in enumerate(puntos_validos):
                    val_f = float(f.subs(p).evalf())
                    
                    mult_str = ", ".join([f"λ_{i+1}={float(p[l].evalf()):.2f}" for i, l in enumerate(l_simbolos)])
                    mu_str = ", ".join([f"μ_{j+1}={float(p[m].evalf()):.2f}" for j, m in enumerate(m_simbolos)])
                    
                    rest_activas_g = [g for g in g_list if abs(float(g.subs(p).evalf())) < 1e-4]
                    total_activas = h_list + rest_activas_g
                    k_l = len(total_activas)
                    
                    if k_l > 0:
                        U_list = [[sp.diff(r, x), sp.diff(r, y)] for r in total_activas]
                        U_mat = sp.Matrix(U_list)
                        HL = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]])
                        H_orlado = sp.zeros(k_l, k_l).row_join(U_mat).col_join(U_mat.T.row_join(HL))
                        det_hor = float(H_orlado.subs(p).det().evalf())
                        
                        tipo_optimo = f"🔵 {objetivo} Condicionado"
                    else:
                        HL = sp.Matrix([[sp.diff(Lx, x), sp.diff(Lx, y)], [sp.diff(Lx, y), sp.diff(Ly, y)]])
                        det_hor = float(HL.subs(p).det().evalf())
                        tipo_optimo = f"🟢 {objetivo} Libre"

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

                # Desglose formal de matrices en LaTeX por punto crítico aprobado
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
        st.error(f"Error general de ejecución: {e}. Valida los campos ingresados.")
