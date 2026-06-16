import streamlit as st
import sympy as sp
import pandas as pd

# Configuración de la página web
st.set_page_config(page_title="Laboratorio de Optimización Avanzada", page_icon="🧮", layout="wide")

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
igualdades_str = st.sidebar.text_input("Restricciones de Igualdad (separadas por ';'):", value="x**2 + y**2 - 25")

st.sidebar.subheader("📐 Restricciones de Desigualdad")
st.sidebar.markdown("Forma estándar: $g(x,y) \le 0$")
desigualdades_str = st.sidebar.text_input("Restricciones de Desigualdad (separadas por ';'):", value="")

st.sidebar.info("**Nota pedagógica:** El sistema ahora factoriza analíticamente las ecuaciones intermedias para abrir sub-casos lógicos (ej: $y=0$ o $\mu_i=k$), evitando omitir raíces válidas.")

calcular = st.sidebar.button("Calcular con Análisis por Factorización", type="primary")

# =========================================================================
# MOTOR ALGEBRAICO CON DESGLOSE DE SUB-CASOS POR FACTORIZACIÓN
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

            # Construcción del Lagrangiano estándar (U. de Piura)
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

            st.header("3. Análisis de Ramificaciones y Sub-casos por Factorización")
            
            num_g = len(g_list)
            puntos_validos = []
            historial_supuestos = []

            # Si no hay restricciones de desigualdad, evaluamos el caso base (K = 0)
            iteraciones = 2**num_g if num_g > 0 else 1

            for combinacion in range(iteraciones):
                supuestos_texto = []
                eqs_sistema = [Lx, Ly] + h_list
                sustituciones_caso = {}
                
                for j in range(num_g):
                    if (combinacion >> j) & 1:
                        eqs_sistema.append(g_list[j])
                        supuestos_texto.append(rf"\mu_{j+1} > 0")
                    else:
                        sustituciones_caso[m_simbolos[j]] = 0
                        supuestos_texto.append(rf"\mu_{j+1} = 0")

                texto_caso_latex = " \quad y \quad ".join(supuestos_texto) if supuestos_texto else "Optimización con Igualdades"
                
                eqs_con_reemplazo = [eq.subs(sustituciones_caso) for eq in eqs_sistema]
                variables_a_resolver = [x, y] + l_simbolos + [m_simbolos[j] for j in range(num_g) if (combinacion >> j) & 1]

                with st.expander(rf"🌿 Evaluando Ramificación: ${texto_caso_latex}$"):
                    st.markdown("**Sistema base tras sustitución de multiplicadores:**")
                    for eq in eqs_con_reemplazo:
                        st.latex(rf"{sp.latex(eq)} = 0")
                    
                    # --- DETECCIÓN PEDAGÓGICA DE FACTORIZACIÓN ---
                    for eq in eqs_con_reemplazo:
                        eq_factored = sp.factor(eq)
                        if isinstance(eq_factored, sp.Mul):
                            st.markdown(f"🔍 **Análisis de Factorización detectado en:** ${sp.latex(eq)} = 0$")
                            st.latex(rf"\implies {sp.latex(eq_factored)} = 0")
                            args = [arg for arg in eq_factored.args if arg.is_number is False]
                            sub_casos_txt = " o ".join([f"${sp.latex(arg)} = 0$" for arg in args])
                            st.info(f"Esto divide el análisis en los siguientes sub-casos independientes: {sub_casos_txt}")

                    try:
                        # Resolver el sistema analíticamente resguardando multiplicidades y soluciones reales
                        sols = sp.solve(eqs_con_reemplazo, variables_a_resolver, dict=True)
                        
                        if not sols:
                            st.error("❌ **Contradicción detectada:** Las ecuaciones son mutuamente incompatibles tras el reemplazo. Este camino queda descartado.")
                            historial_supuestos.append({
                                "Supuesto Evaluado": f"${texto_caso_latex}$",
                                "Estatus Algebraico": "❌ Contradicción",
                                "Resultado": "Descartado"
                            })
                            continue
                        
                        st.success(f"📊 **Sistema analizado con éxito.** Se encontraron {len(sols)} soluciones de ramificación:")
                        
                        caminos_validos_en_este_supuesto = 0
                        
                        for idx_s, sol in enumerate(sols):
                            for k, v in sustituciones_caso.items():
                                sol[k] = sp.simplify(v)
                            
                            # Validar que existan soluciones en el plano real
                            if sol[x].is_real and sol[y].is_real:
                                cumple_kkt = True
                                motivos_descarte = []

                                # Mostrar rastro de la sustitución al usuario de forma clara
                                st.markdown(f"**Sub-caso {idx_s + 1}:** Coordenada matemática hallada: $x = {sp.latex(sol[x])}$, $y = {sp.latex(sol[y])}$")

                                # Validación de restricciones de desigualdad g(x,y) <= 0
                                for idx_g, g_expr in enumerate(g_list):
                                    val_g = float(g_expr.subs(sol).evalf())
                                    if val_g > 1e-4:
                                        cumple_kkt = False
                                        motivos_descarte.append(f"g_{idx_g+1} fuera de región factible ({val_g:.2f} > 0)")

                                # Validación de signos de multiplicadores mu >= 0
                                for mu_sym in m_simbolos:
                                    if mu_sym in sol:
                                        val_mu = float(sol[mu_sym].evalf())
                                        if val_mu < -1e-4:
                                            cumple_kkt = False
                                            motivos_descarte.append(rf"Multiplicador con signo incorrecto ({sp.latex(mu_sym)} = {val_mu:.2f} < 0)")

                                if cumple_kkt:
                                    sol_copia = sol.copy()
                                    if sol_copia not in puntos_validos:
                                        puntos_validos.append(sol_copia)
                                    caminos_validos_en_este_supuesto += 1
                                    st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;🎯 *Resultado:* Satisface completamente los criterios de optimalidad.")
                                else:
                                    st.markdown(rf"&nbsp;&nbsp;&nbsp;&nbsp;⚠️ *Resultado:* Raíz descartada por: {', '.join(motivos_descarte)}.")
                        
                        if caminos_validos_en_este_supuesto > 0:
                            historial_supuestos.append({
                                "Supuesto Evaluado": f"${texto_caso_latex}$",
                                "Estatus Algebraico": "✅ Consistente (Factorizado)",
                                "Resultado": f"Aporta {caminos_validos_en_este_supuesto} punto(s) crítico(s)"
                            })
                        else:
                            historial_supuestos.append({
                                "Supuesto Evaluado": f"${texto_caso_latex}$",
                                "Estatus Algebraico": "✅ Consistente",
                                "Resultado": "Descartado por condiciones de frontera/signo"
                            })

                    except Exception as e:
                        st.error(f"Error al descomponer el sistema analítico: {e}")

            # Tabla resumen
            st.subheader("📋 Tabla Resumen del Proceso Analítico")
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
        st.error(f"Error general en el procesamiento analítico: {e}. Asegúrate de escribir correctamente la sintaxis algebraica.")
