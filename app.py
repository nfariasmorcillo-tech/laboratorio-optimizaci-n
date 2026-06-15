import streamlit as st
import sympy as sp
import pandas as pd

# Configuración de la página web
st.set_page_config(page_title="Laboratorio de Optimización", page_icon="🧮", layout="wide")

st.title("🧮 Laboratorio de Optimización Matemática")
st.markdown("Herramienta pedagógica para el análisis de óptimos locales con y sin restricciones.")

# Barra lateral para el ingreso de datos
st.sidebar.header("📥 Configuración del Ejercicio")

tipo_opt = st.sidebar.radio(
    "Selecciona el tipo de optimización:",
    ("Optimización Libre (Sin restricciones)", "Optimización Condicionada (Con restricciones)")
)

funcion_str = st.sidebar.text_input("Función Objetivo f(x,y):", value="x**2 + y**2 - 4*x - 6*y")
variables_str = st.sidebar.text_input("Variables (separadas por coma):", value="x, y")

restriccion_str = ""
if tipo_opt == "Optimización Condicionada (Con restricciones)":
    restriccion_str = st.sidebar.text_input("Restricción g(x,y) = 0:", value="x + y - 4")

st.sidebar.info("**Nota de sintaxis:**\n* Usar `*` para multiplicar (ej: `4*x`).\n* Usar `**` para potencias (ej: `x**2`).")

# Botón para ejecutar el cálculo
calcular = st.sidebar.button("Calcular Óptimos", type="primary")

if calcular:
    try:
        # Procesar variables
        vars_list = [sp.Symbol(v.strip()) for v in variables_str.split(',') if v.strip()]
        if len(vars_list) != 2:
            st.error("Por favor, ingresa exactamente dos variables libres (ej: x, y).")
        else:
            x, y = vars_list[0], vars_list[1]
            f = sp.sympify(funcion_str)
            
            # --- SECCIÓN: ENUNCIADO ---
            col1, col2 = st.columns(2)
            with col1:
                st.latex(rf"\text{{Función Objetivo: }} f({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(f)}")
            with col2:
                if tipo_opt == "Optimización Condicionada (Con restricciones)":
                    g = sp.sympify(restriccion_str)
                    st.latex(rf"\text{{Restricción: }} g({sp.latex(x)}, {sp.latex(y)}) = {sp.latex(g)} = 0")

            st.divider()

            # ==========================================
            # CASO 1: OPTIMIZACIÓN LIBRE
            # ==========================================
            if tipo_opt == "Optimización Libre (Sin restricciones)":
                st.header("1. Primeras Derivadas y Sistema de Ecuaciones")
                fx = sp.diff(f, x)
                fy = sp.diff(f, y)
                
                c1, c2 = st.columns(2)
                c1.latex(rf"f_x = \frac{{\partial f}}{{\partial x}} = {sp.latex(fx)}")
                c2.latex(rf"f_y = \frac{{\partial f}}{{\partial y}} = {sp.latex(fy)}")
                
                st.subheader("Sistema a resolver para puntos estacionarios:")
                sistema_libre = r"\begin{cases} " + sp.latex(fx) + r" = 0 \\ " + sp.latex(fy) + r" = 0 \end{cases}"
                st.latex(sistema_libre)
                
                # Resolver
                puntos = sp.solve([fx, fy], (x, y), dict=True)
                st.metric(label="Cantidad de puntos críticos encontrados", value=len(puntos))
                
                st.header("2. Segundas Derivadas y Matriz Hessiana")
                fxx = sp.diff(fx, x)
                fyy = sp.diff(fy, y)
                fxy = sp.diff(fx, y)
                H_gen = sp.Matrix([[fxx, fxy], [fxy, fyy]])
                
                st.latex(rf"\text{{Segunda derivada respecto a x: }} f_{{xx}} = {sp.latex(fxx)}")
                st.latex(rf"\text{{Matriz Hessiana (H): }} {sp.latex(H_gen)}")
                st.latex(rf"\text{{Hessiana (Determinante |H| o }} H_2\text{{): }} {sp.latex(H_gen.det())}")
                
                if len(puntos) > 0:
                    st.header("3. Tabla de Clasificación de Óptimos")
                    
                    datos_tabla = []
                    for i, p in enumerate(puntos):
                        if p[x].evalf().is_imaginary or p[y].evalf().is_imaginary:
                            continue
                        det_p = H_gen.det().subs(p)
                        fxx_p = fxx.subs(p)
                        val_f = f.subs(p)
                        
                        if det_p > 0:
                            tipo = "🟢 Mínimo Local" if fxx_p > 0 else "🔵 Máximo Local"
                        elif det_p < 0:
                            tipo = "🔴 Punto Silla"
                        else:
                            tipo = "⚪ No decide"
                        
                        datos_tabla.append({
                            "ID": f"P{i+1}",
                            "Punto (x, y) Analítico": f"({sp.latex(p[x])}, {sp.latex(p[y])})",
                            "Punto (x, y) Decimal": f"({float(p[x].evalf()):.2f}, {float(p[y].evalf()):.2f})",
                            "f_xx (H1)": str(fxx_p),
                            "|H| (H2)": f"{str(det_p)} ({float(det_p.evalf()):.2f})",
                            "Tipo de Óptimo": tipo,
                            "Valor f(x,y)": f"{str(val_f)} ({float(val_f.evalf()):.2f})"
                        })
                    
                    df = pd.DataFrame(datos_tabla)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No se hallaron puntos críticos reales.")

            # ==========================================
            # CASO 2: OPTIMIZACIÓN CON RESTRICCIONES
            # ==========================================
            else:
                lam = sp.Symbol('λ')
                L = f - lam * g
                
                st.header("1. Función del Lagrangiano")
                st.latex(rf"\mathcal{{L}}({sp.latex(x)}, {sp.latex(y)}, \lambda) = f({sp.latex(x)},{sp.latex(y)}) - \lambda \cdot g({sp.latex(x)},{sp.latex(y)}) = {sp.latex(L)}")
                
                st.header("2. Primeras Derivadas de la Restricción y el Lagrangiano")
                gx = sp.diff(g, x)
                gy = sp.diff(g, y)
                Lx = sp.diff(L, x)
                Ly = sp.diff(L, y)
                Llam = sp.diff(L, lam)
                
                c1, c2 = st.columns(2)
                c1.latex(rf"g_x = \frac{{\partial g}}{{\partial x}} = {sp.latex(gx)}")
                c2.latex(rf"g_y = \frac{{\partial g}}{{\partial y}} = {sp.latex(gy)}")
                
                st.subheader("Sistema a resolver (Condiciones de Primer Orden):")
                sistema_condicionado = r"\begin{cases} \mathcal{L}_x = " + sp.latex(Lx) + r" = 0 \\ \mathcal{L}_y = " + sp.latex(Ly) + r" = 0 \\ \mathcal{L}_\lambda = " + sp.latex(Llam) + r" = 0 \end{cases}"
                st.latex(sistema_condicionado)
                
                # Resolver
                puntos = sp.solve([Lx, Ly, Llam], (x, y, lam), dict=True)
                st.metric(label="Cantidad de puntos críticos encontrados", value=len(puntos))
                
                st.header("3. Derivadas de Segundo Orden del Lagrangiano")
                Lxx = sp.diff(Lx, x)
                Lyy = sp.diff(Ly, y)
                Lxy = sp.diff(Lx, y)
                
                st.latex(rf"\mathcal{{L}}_{{xx}} = {sp.latex(Lxx)} \quad | \quad \mathcal{{L}}_{{yy}} = {sp.latex(Lyy)} \quad | \quad \mathcal{{L}}_{{xy}} = {sp.latex(Lxy)}")
                
                st.header("4. Matriz del Hessiano Orlado (HOR)")
                HOR = sp.Matrix([[0, gx, gy], [gx, Lxx, Lxy], [gy, Lxy, Lyy]])
                st.latex(rf"\text{{HOR}} = {sp.latex(HOR)}")
                st.latex(rf"|\text{{HOR}}| = {sp.latex(HOR.det())}")
                
                if len(puntos) > 0:
                    st.header("5. Resultados y Criterio del Hessiano Orlado")
                    
                    datos_tabla = []
                    for i, p in enumerate(puntos):
                        if p[x].evalf().is_imaginary or p[y].evalf().is_imaginary:
                            continue
                        det_p = HOR.det().subs(p)
                        hor_eval = HOR.subs(p)
                        val_f = f.subs({x: p[x], y: p[y]})
                        
                        tipo = "🔵 Máximo condicionado" if det_p > 0 else "🟢 Mínimo condicionado" if det_p < 0 else "⚪ No decide"
                        
                        # Convertimos la matriz HOR evaluada a un formato de texto compacto tipo MATLAB/Octave para la tabla
                        matriz_texto = str(hor_eval.tolist()).replace('[', '').replace(']', '').replace(', ', ' ')
                        # Formato amigable: filas separadas por barras
                        filas_m = [str(hor_eval.row(r).tolist()).replace('[','').replace(']','') for r in range(3)]
                        matriz_compacta = " | ".join(filas_m)

                        datos_tabla.append({
                            "ID": f"P{i+1}",
                            "Punto (x, y)": f"({float(p[x].evalf()):.2f}, {float(p[y].evalf()):.2f})",
                            "Valor λ": f"{float(p[lam].evalf()):.2f}",
                            "Matriz HOR resumen": matriz_compacta,
                            "|HOR| (Det)": f"{float(det_p.evalf()):.2f}",
                            "Tipo de Óptimo": tipo,
                            "Valor f(x,y)": f"{float(val_f.evalf()):.2f}"
                        })
                    
                    df = pd.DataFrame(datos_tabla)
                    
                    # Mostramos la tabla perfectamente estructurada
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # --- DESGLES PEDAGÓGICO DE MATRICES ---
                    st.subheader("🔍 Desglose matemático por cada punto:")
                    for i, p in enumerate(puntos):
                        if p[x].evalf().is_imaginary or p[y].evalf().is_imaginary:
                            continue
                        det_p = HOR.det().subs(p)
                        hor_eval = HOR.subs(p)
                        
                        st.markdown(f"**Análisis Detallado del Punto P{i+1}** en $({sp.latex(p[x])}, {sp.latex(p[y])})$:")
                        col_m1, col_m2 = st.columns(2)
                        col_m1.latex(rf"\text{{HOR}}_{{P{i+1}}} = {sp.latex(hor_eval)}")
                        col_m2.latex(rf"|\text{{HOR}}_{{P{i+1}}}| = {sp.latex(det_p)}")
                        st.caption("---")
                else:
                    st.warning("No se hallaron puntos críticos analíticos para el Lagrangiano.")
                    
    except Exception as e:
        st.error(f"Error en la entrada matemática: {e}. Revisa si faltan operadores `*` o `**`.")
