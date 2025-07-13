# app.py
import streamlit as st
import pandas as pd
from services.predictor_service import PredictorService
from utils.history_logger import load_history


st.set_page_config(page_title="Sistema Inteligente de Compras", layout="wide")
st.title("🧠 Sistema Inteligente de Predicción de Compras")
st.markdown("Este MVP predice la demanda de productos usando Machine Learning y principios SOLID.")

model_choice = st.selectbox("🧠 Elegir modelo de predicción", ["xgboost", "random_forest"])
tab1, tab2 = st.tabs(["📊 Entrenamiento del modelo", "🔮 Predicción"])


# Ruta al dataset generado
DATA_PATH = "ventas.csv"



# Instanciar el servicio
service = PredictorService(DATA_PATH, model_choice)

with tab1:
    st.info(f"🔍 Modelo seleccionado: **{model_choice}**")
    if st.button("🔧 Entrenar modelo"):
        with st.spinner("Entrenando modelo..."):
            metrics = service.run_training()
            st.success("✅ Modelo entrenado con éxito")
            st.metric("MAE", round(metrics["MAE"], 2))
            st.metric("RMSE", round(metrics["RMSE"], 2))

        with st.expander("📜 Ver historial de entrenamientos"):
            history = load_history()
            if history:
                st.write("📈 Evolución de métricas")
                hist_df = pd.DataFrame(history)
                st.dataframe(hist_df)
                st.line_chart(hist_df.set_index("fecha")[["MAE", "RMSE"]])
            else:
                st.info("Aún no hay historial registrado.")

with tab2:
    st.info(f"📌 Predicciones generadas con modelo: **{model_choice}**")
    if st.button("📈 Predecir demanda"):
        with st.spinner("Generando predicciones..."):
            predictions = service.predict_from_file()
            df = pd.read_csv(DATA_PATH)
            df['prediccion'] = predictions
            st.dataframe(df[['fecha', 'producto_id', 'cantidad', 'prediccion']])
            
            # Mostrar gráfico comparativo
            st.subheader("📊 Comparación visual de cantidad vs predicción")
            chart_df = df[['fecha', 'cantidad', 'prediccion']].copy()
            chart_df = chart_df.groupby('fecha').sum().reset_index()
            chart_df = chart_df.sort_values('fecha')

            st.line_chart(chart_df.set_index('fecha'))

            # Comparación por producto
            st.subheader("🧮 Promedio por producto: cantidad vs predicción")

            prod_df = df[['producto_id', 'cantidad', 'prediccion']].copy()
            prod_df = prod_df.groupby('producto_id').mean().reset_index()

            st.bar_chart(prod_df.set_index('producto_id'))

            # Error absoluto medio por producto
            st.subheader("📉 Error absoluto medio por producto")

            df['error_abs'] = abs(df['cantidad'] - df['prediccion'])
            error_df = df.groupby('producto_id')['error_abs'].mean().reset_index()
            error_df = error_df.sort_values('error_abs', ascending=False)

            st.bar_chart(error_df.set_index('producto_id'))

            # Mostrar un 🔹 Scatter plot real vs predicho
            st.subheader("🔹 Gráfico de dispersión: Real vs Predicho")
            st.scatter_chart(df[['cantidad', 'prediccion']].rename(columns={'cantidad': 'Real', 'prediccion': 'Predicho'}))

            # # 🔹 Heatmap por semana y producto
            # st.subheader("🔹 Heatmap de ventas por semana y producto")
            # heatmap_df = df.pivot_table(index='semana', columns='producto_id', values='cantidad', aggfunc='sum')
            # st.write(heatmap_df)
            # st.heatmap(heatmap_df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Cantidad Vendida'})
            # st.success("✅ Predicciones generadas con éxito")
            st.info(f"📌Predicciones generadas para el cliente:")

            st.subheader("💬 Recomendaciones por producto")
            for idx, row in df.groupby('producto_id').mean(numeric_only=True).reset_index().iterrows():
                prod = row['producto_id']
                pred = row['prediccion']
                mensaje = f"🛒 Producto {prod}: predicción de {pred:.1f} unidades. "
                if pred > 30:
                    mensaje += "Alta demanda esperada. ✔️ Comprar más unidades."
                elif pred > 10:
                    mensaje += "Demanda moderada. 🟡 Revisar stock antes de comprar."
                else:
                    mensaje += "Baja demanda prevista. ❌ No se recomienda comprar por ahora."
                st.markdown(f"- {mensaje}")
            
            st.subheader("🚨 Alertas inteligentes de compra")

            for prod_id in df['producto_id'].unique():
                sub_df = df[df['producto_id'] == prod_id]
                promedio_stock = sub_df['stock'].mean()
                promedio_prediccion = sub_df['prediccion'].mean()

                if promedio_prediccion > 30 and promedio_stock < 40:
                    mensaje = (
                        f"📦 Producto **{prod_id}**:\n"
                        f"🔺 Stock bajo (~{int(promedio_stock)} unidades)\n"
                        f"🔮 Predicción alta (~{int(promedio_prediccion)} unidades)\n"
                        f"✅ *Recomendación: Comprar pronto*\n"
                    )
                    st.markdown(mensaje)
                elif promedio_prediccion < 10:
                    st.markdown(f"❌ Producto **{prod_id}**: demanda baja (~{int(promedio_prediccion)}) → No comprar")
            #  Ranking de productos más demandados (por predicción)
            st.subheader("🏆 Ranking de productos más demandados")
            ranking_df = df.groupby('producto_id')['prediccion'].mean().reset_index()
            ranking_df = ranking_df.sort_values('prediccion', ascending=False)
            st.write(ranking_df)
        st.subheader("🛒 Productos recomendados para esta semana")

        recomendaciones = df.groupby('producto_id')[['prediccion', 'stock']].mean().reset_index()
        recomendaciones['prioridad'] = recomendaciones['prediccion'] - recomendaciones['stock']
        recomendaciones = recomendaciones.sort_values(by='prioridad', ascending=False)

        st.dataframe(recomendaciones[['producto_id', 'prediccion', 'stock', 'prioridad']].head(10))

        st.caption("🔎 Se recomienda revisar los productos con mayor prioridad (demanda mayor que el stock disponible ). "
           "Esta tabla muestra los productos que deberías considerar comprar esta semana, ordenados por prioridad.")

        st.subheader("📈 Tendencia semanal de demanda estimada")

        df['fecha'] = pd.to_datetime(df['fecha'])
        df['semana'] = df['fecha'].dt.to_period('W').astype(str)

        trend = df.groupby('semana')['prediccion'].sum().reset_index()
        st.line_chart(trend.set_index('semana'))

        st.caption("📆 Ayuda a anticiparse a semanas de alta o baja demanda.")

        st.subheader("❌ Productos con baja demanda")

        bajos = df.groupby('producto_id')['prediccion'].mean().reset_index()
        bajos = bajos[bajos['prediccion'] < 10].sort_values(by='prediccion')

        st.dataframe(bajos)

        st.caption("⛔ Evitar compras innecesarias de productos con poca salida.")

        st.markdown("### 🛍️ Resumen para esta semana:")

        # Clasificar productos por prioridad
        def clasificar_prioridad(row):
            if row['prioridad'] > 30:
                return "✔️ Alta prioridad"
            elif row['prioridad'] > 10:
                return "🟡 Media prioridad"
            else:
                return "❌ Baja prioridad"

        recomendaciones['grupo_prioridad'] = recomendaciones.apply(clasificar_prioridad, axis=1)
        grupo_count = recomendaciones['grupo_prioridad'].value_counts()

        altos = grupo_count.get("✔️ Alta prioridad", 0)
        medios = grupo_count.get("🟡 Media prioridad", 0)
        bajos = grupo_count.get("❌ Baja prioridad", 0)

        st.success(f"✔️ {altos} productos con demanda alta → Comprar pronto")
        st.warning(f"🟡 {medios} productos con demanda media → Revisar stock manualmente")
        st.info(f"❌ {bajos} productos con demanda baja → No comprar esta semana")         


    with st.expander("📄 Ver dataset original"):
        df = pd.read_csv(DATA_PATH)
        st.dataframe(df.head())

st.sidebar.header("Información del Proyecto")
st.sidebar.markdown("""
Este proyecto utiliza un modelo de XGBoost para predecir la demanda de productos en tiendas, basado en datos históricos de ventas. Implementa principios SOLID para asegurar un código limpio y mantenible.
""")
st.sidebar.markdown("""### Principios SOLID aplicados:
- **Single Responsibility Principle (SRP)**: Cada clase tiene una única responsabilidad.
- **Open/Closed Principle (OCP)**: El modelo puede ser extendido sin modificar su código.
- **Liskov Substitution Principle (LSP)**: Se puede sustituir el modelo por otro que implemente la misma interfaz.
- **Interface Segregation Principle (ISP)**: Se define una interfaz clara para los modelos.
- **Dependency Inversion Principle (DIP)**: El servicio de predicción depende de abstracciones, no de implementaciones concretas.
""")