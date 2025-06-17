import sys
import streamlit as st
from io import StringIO
from generic_functions import (generar_imagen_s1, 
                               generate_mask, generar_imagen_s2, calcular_porcentaje_cambio, porcentaje_por_indice, 
                               obtener_ubicacion, generar_analisis_ollama3, test_model)
from PIL import Image, ImageEnhance
import time
import os

st.title("Visualizador de Detección de Cambios con imágenes SAR")

# Entrada de coordenadas y fechas
st.sidebar.header("Parámetros de Entrada")

longitude = st.sidebar.number_input("Longitud", value=-78.527264, format="%.6f")
latitude = st.sidebar.number_input("Latitud", value=-3.106770, format="%.6f")
fecha_objetivo1 = st.sidebar.date_input("Fecha 1", value=None)
fecha_objetivo2 = st.sidebar.date_input("Fecha 2", value=None)

# Botón para generar imágenes
if st.sidebar.button("Generar imágenes"):
    if not fecha_objetivo1 or not fecha_objetivo2:
        st.warning("Por favor selecciona ambas fechas.")
    else:
        # Convertir fechas a string
        fecha1_str = fecha_objetivo1.strftime("%Y-%m-%d")
        fecha2_str = fecha_objetivo2.strftime("%Y-%m-%d")
    
        # Mostrar ubicación
        ubicacion = obtener_ubicacion(latitude, longitude)
        if isinstance(ubicacion, dict):
            st.sidebar.success("Ubicación obtenida con éxito.")
            st.sidebar.write(f"Dirección: {ubicacion['Dirección']}")
            st.sidebar.write(f"País: {ubicacion['País']}")
            st.sidebar.write(f"Estado: {ubicacion['Estado']}")
            st.sidebar.write(f"Ciudad: {ubicacion['Ciudad']}")
        else:
            st.sidebar.error(ubicacion)
        with st.spinner("Generando imágenes..."):
            # Sentinel-2
            img_sentinel2A = generar_imagen_s2(longitude, latitude, fecha1_str, "Sentinel-2/A")
            img_sentinel2B = generar_imagen_s2(longitude, latitude, fecha2_str, "Sentinel-2/B")

            # Sentinel-1 - RVI
            image_before = generar_imagen_s1(longitude, latitude, fecha1_str, "BAN/data/LEVIR-CD/test/A")
            image_after = generar_imagen_s1(longitude, latitude, fecha2_str, "BAN/data/LEVIR-CD/test/B")
            mask = generate_mask(image_before, image_after)

            # Tabs
            st.markdown(
                """
                Las imágenes Sentinel-2 se muestran para ofrecer una visión más realista del área seleccionada; sin embargo, pueden aparecer completamente blancas (cubiertas de nubes) 
                debido a la falta de disponibilidad de imágenes ópticas en presencia de nubosidad. A continuación, se presentan también las imágenes Sentinel-1, que serán procesadas por el modelo, junto con la máscara de cambio generada.
                """
            )
            tab1, tab2 = st.tabs(["Sentinel-2", "Sentinel-1"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                if img_sentinel2A is not None and img_sentinel2B is not None:
                    col1.image(img_sentinel2A, caption="Antes", use_column_width=True)
                    col2.image(img_sentinel2B, caption="Después", use_column_width=True)
                else:
                    st.warning("No se pudieron generar las imágenes Sentinel-2.")

            with tab2:
                col3, col4, col5 = st.columns(3)
                if image_before is not None and image_after is not None:
                    col3.image(image_before, caption="Antes", use_column_width=True)
                    col4.image(image_after, caption="Después", use_column_width=True)
                else:
                    st.warning("No se pudieron generar las imágenes Sentinel-1.")

        # Ejecutar el modelo después de mostrar las imágenes
        with st.spinner("Detectando cambios..."):
            test_model()
            # Mostrar resultados del modelo en la tercera columna de cada tab
            with tab2:
                if image_before is not None and image_after is not None:
                    if hasattr(image_before, 'filename'):
                        filename = os.path.basename(image_before.filename)
                    elif isinstance(image_before, str):
                        filename = os.path.basename(image_before)
                    else:
                        filename = None
                    if filename:
                        result_path = os.path.join("BAN/resultados/vis_data/vis_image", filename)
                        if os.path.exists(result_path):
                            col5.image(result_path, caption="Resultado del modelo", use_column_width=True)
                        else:
                            col5.warning("No se encontró la imagen de resultado del modelo.")
                    else:
                        col5.warning("No se pudo determinar el nombre del archivo para mostrar el resultado.")

        with st.spinner("Calculando porcentaje de cambio..."):
            porcentaje_cambio = calcular_porcentaje_cambio(result_path)

            # Mostrar porcentaje de cambio como barra de progreso
            st.subheader("Porcentaje de Cambio")
            st.progress(porcentaje_cambio/100)
            st.write(f"Porcentaje de cambio: {porcentaje_cambio:.2f}%")

        with st.spinner("Generando porcentajes de cambio por índice..."):
            # Generar porcentajes de cambio en base al indice de vegetación
            porcentaje_cambio_indice, total = porcentaje_por_indice(image_before, image_after)

            # Mostrar porcentajes de cambio por índice
            st.subheader("Porcentaje de Cambio por Índice")
            for (g1, g2), count in porcentaje_cambio_indice.items():
                if g1 != g2:
                    porcentaje = count / total
                    st.write(f"{g1} -> {g2}: {porcentaje*100:.2f}%")
                    st.progress(min(max(porcentaje, 0.0), 1.0))

        with st.spinner("Generando análisis por modelo Ollama3..."):
            # Generar análisis con Ollama3
            analisis = generar_analisis_ollama3(ubicacion, porcentaje_cambio_indice)
    
            # Mostrar análisis
            st.subheader("Análisis de Cambios Generado por Ollama3")
            st.write(analisis)
