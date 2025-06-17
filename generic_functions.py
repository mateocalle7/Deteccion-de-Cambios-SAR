from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
import requests
import json
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from collections import defaultdict
import os
import cv2
import numpy as np
from config import client_id, client_secret
import time
import math
import subprocess

def generar_poligono(longitude, latitude):
    """
    Genera un polígono cuadrado de 1.05 km², se define por las coordenadas centrales.
    """
    # Parámetros
    area_km2 = 1.05  # Área en km²
    lado_km = math.sqrt(area_km2)  # Lado del cuadrado en km
    lado_grados_lat = lado_km / 111.32 

    # Ajuste longitudinal según la latitud
    lat_rad = math.radians(latitude)
    km_por_grado_lon = 111.32 * math.cos(lat_rad)
    lado_grados_lon = lado_km / km_por_grado_lon

    # Desplazamiento desde el centro a cada lado
    delta_lat = lado_grados_lat / 2
    delta_lon = lado_grados_lon / 2

    # Calcular las esquinas del cuadrado
    lat_nw = latitude + delta_lat
    lon_nw = longitude - delta_lon
    lat_ne = latitude + delta_lat
    lon_ne = longitude + delta_lon
    lat_sw = latitude - delta_lat
    lon_sw = longitude - delta_lon
    lat_se = latitude - delta_lat
    lon_se = longitude + delta_lon

    # Crear la lista de coordenadas (cerrando el polígono)
    coordenadas = [
        [
            [lon_nw, lat_nw],  # NW
            [lon_ne, lat_ne],  # NE
            [lon_se, lat_se],  # SE
            [lon_sw, lat_sw],  # SW
            [lon_nw, lat_nw]   # Cierre
        ]
    ]

    # Retornar coordenadas
    return coordenadas

def obtener_ubicacion(latitud, longitud):
    """
    Obtiene la ubicación descriptiva (dirección, país, estado, ciudad) a partir de coordenadas geográficas.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?lat={latitud}&lon={longitud}&format=json"
    
    # User-Agent personalizado 
    headers = {
        'User-Agent': 'MiApp/1.0'
    }
    
    try:
        # Pausa para cumplir con políticas de uso (1 solicitud por segundo)
        time.sleep(1)
        
        response = requests.get(url, headers=headers)
        
        # Verificar respuesta
        if response.status_code == 403:
            return "Error 403: Acceso denegado. Espera unos minutos o usa un User-Agent válido."
        elif response.status_code != 200:
            return f"Error HTTP {response.status_code}: {response.reason}"
        
        data = response.json()
        
        if "error" in data:
            return "Ubicación no encontrada."
        
        # Extraer datos
        direccion = data.get("display_name", "Desconocido")
        address = data.get("address", {})
        resultado = {
            "Dirección": direccion,
            "País": address.get("country", "Desconocido"),
            "Estado": address.get("state", "Desconocido"),
            "Ciudad": address.get("city", address.get("town", address.get("village", "Desconocido")))
        }
        return resultado
    
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: {str(e)}"
    except Exception as e:
        return f"Error inesperado: {str(e)}"

def generar_rango_fechas(fecha_objetivo_str):
    """
    Genera un rango de fechas en formato ISO 8601 de +/- 5 días alrededor de una fecha objetivo.
    """
    # Convertir la fecha objetivo a un objeto datetime
    fecha_objetivo = datetime.strptime(fecha_objetivo_str, "%Y-%m-%d")

    # Definir un rango de +/- 5 días alrededor de la fecha objetivo
    delta = timedelta(days=5)
    fecha_inicio = (fecha_objetivo - delta).strftime("%Y-%m-%dT00:00:00Z")
    fecha_fin = (fecha_objetivo + delta).strftime("%Y-%m-%dT23:59:59Z")

    return fecha_inicio, fecha_fin

def generar_imagen_s1(longitude, latitude, fecha_objetivo_str, folder):
    """
    Descarga una imagen Sentinel-1 SAR procesada para el polígono centrado en las coordenadas y rango de fechas dados.
    El polígono se define por las coordenadas centrales.
    """
    # Obtener token de acceso
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    token = oauth.fetch_token(
        token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )

    # Configuración de la solicitud de procesamiento
    url = "https://services.sentinel-hub.com/api/v1/process"

    coordenadas = generar_poligono(longitude, latitude)
    fecha_inicio, fecha_fin = generar_rango_fechas(fecha_objetivo_str)

    request_payload = {
        "input": {
            "bounds": {
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordenadas,
                    "properties": {
                        "crs": "http://www.opengis.net/def/crs/EPSG/0/3857"
                    }
                }
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {
                            "from": fecha_inicio,
                            "to": fecha_fin
                        },
                        "acquisitionMode": "IW",
                        "polarization": "DV", 
                        "orbitDirection ": "ASCENDING"
                    },
                    "processing": {
                        "orthorectify": "true",
                        "demInstance": "COPERNICUS_30",
                        "speckleFilter": {
                            "type": "LEE",
                            "windowSizeX": 7,
                            "windowSizeY": 7
                        },
                        "upsampling": "BICUBIC"
                    }
                }
            ]
        },
        "output": {
            "width": 1024,
            "height": 1024,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    }
                }
            ]
        }
    }

    evalscript = """
                // Radar vegetation index for Sentinel-1 (RVI4S1) SAR data
                // Institute: MRSLab, Indian Institute of Technology Bombay, India
                // Reference: Bhogapurapu et al., 2022

                function setup() {
                return {
                    input: ["VV", "VH", "dataMask"],
                    output: { bands: 4 }
                };
                }

                // Rampa mejorada: más niveles de marrón y verde, sin blanco
                const ramp = [
                [0.0,  0x3e2615], // marrón muy oscuro – suelo desnudo
                [0.15, 0x5c3a21], // marrón oscuro
                [0.3,  0x8b5a2b], // marrón medio
                [0.45, 0xa67c52], // marrón claro

                [0.6,  0x85b96e], // verde pálido – poca vegetación
                [0.75, 0x5ca450], // verde medio – vegetación moderada
                [0.9,  0x388c35], // verde intenso
                [1.0,  0x276419]  // verde muy oscuro – vegetación densa
                ];

                const visualizer = new ColorRampVisualizer(ramp);

                function evaluatePixel(samples) {
                let VV = samples.VV;
                let VH = samples.VH;
                let dataMask = samples.dataMask;

                let ratio = VH / VV;

                // === Detección de agua ===
                let isWater = (VV < 0.025 && VH < 0.015) || (ratio < 0.045 && VV < 0.04);
                let isPossiblyNoise = VV > 0.04 && VH > 0.02 && ratio > 0.1;
                if (isWater && isPossiblyNoise) {
                    isWater = false;
                }

                if (isWater) {
                    return [0, 0, 1, dataMask]; // azul puro – agua
                }

                // Cálculo del índice de vegetación basado en ratio
                let q = ratio;
                let value = (q * (q + 3)) / Math.pow(q + 1, 2);
                value = Math.min(1, Math.max(0, value)); // asegurarse de que esté entre 0 y 1

                let imgVals = visualizer.process(value);
                return imgVals.concat(dataMask);
                }
                """

    # Preparar los archivos para la solicitud multipart
    files = {
        'request': (None, json.dumps(request_payload), 'application/json'),
        'evalscript': (None, evalscript)
    }

    # Obtener el siguiente nombre de archivo disponible en la carpeta
    # Crear el directorio si no existe
    os.makedirs(folder, exist_ok=True)
    existing_files = sorted([int(f.split(".")[0]) for f in os.listdir(folder) if f.endswith(".png") and f.split(".")[0].isdigit()])
    next_number = (existing_files[-1] + 1) if existing_files else 1
    filename = os.path.join(folder, f"{next_number}.png")

    response = oauth.post(url, files=files)

    # Procesar la respuesta
    if response.status_code == 200:
        print("Solicitud exitosa")
        # Guardar la imagen resultante
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Imagen guardada como {filename}")
        # retornar el path de la imagen
        return filename
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print(response.text)
        return None
    
def generate_mask(image_before, image_after):
    """
    Genera una máscara binaria de cambio entre dos imágenes, resaltando las áreas con diferencias significativas.
    """
    img_A = cv2.imread(image_before)
    img_B = cv2.imread(image_after)
    if img_A is None or img_B is None:
        print(f"Error al cargar imágenes: {image_before}, {image_after}")
        return
    gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY)
    gray_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray_A, gray_B)
    _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray_A)

    for cnt in contours:
        if cv2.contourArea(cnt) > 400:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Crear directorio de salida si no existe
    os.makedirs("BAN/data/LEVIR-CD/test/label", exist_ok=True)

    # Nombre de archivo (usamos el de la imagen A)
    filename = os.path.basename(image_before)
    save_path = os.path.join("BAN/data/LEVIR-CD/test/label", filename)

    # Guardar máscara
    cv2.imwrite(save_path, mask)
    print(f"Máscara guardada en: {save_path}")
    return save_path

def generar_imagen_s2(longitude, latitude, fecha_objetivo_str, folder):
    """
    Descarga una imagen Sentinel-2 óptica procesada para el polígono centrado en las coordenadas y rango de fechas dados.
    El polígono se define por las coordenadas centrales.
    """
    # Obtener token de acceso
    client = BackendApplicationClient(client_id=client_id)
    oauth = OAuth2Session(client=client)

    token = oauth.fetch_token(
        token_url='https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token',
        client_secret=client_secret,
        include_client_id=True
    )

    # Configuración de la solicitud de procesamiento
    url = "https://services.sentinel-hub.com/api/v1/process"

    coordenadas = generar_poligono(longitude, latitude)
    fecha_inicio, fecha_fin = generar_rango_fechas(fecha_objetivo_str)
    
    request_payload = {
        "input": {
            "bounds": {
                "properties": {
                    "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coordenadas,
                }
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": fecha_inicio,
                            "to": fecha_fin
                        }
                        # ,"maxCloudCoverage": 30
                    }
                }
            ]
        },
        "output": {
            "width": 1024,
            "height": 1024,
            "responses": [
                {
                    "identifier": "default",
                    "format": {
                        "type": "image/png"
                    }
                }
            ]
        }
    }

    evalscript = """
    //VERSION=3

    function setup() {
    return {
        input: ["B02", "B03", "B04"],
        output: {
        bands: 3,
        sampleType: "AUTO" // default value - scales the output values from [0,1] to [0,255].
        }
    }
    }

    function evaluatePixel(sample) {
    return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02]
    }
    """

    # Preparar los archivos para la solicitud multipart
    files = {
        'request': (None, json.dumps(request_payload), 'application/json'),
        'evalscript': (None, evalscript)
    }
    
    # Crear el directorio si no existe
    os.makedirs(folder, exist_ok=True)
    # Obtener el siguiente nombre de archivo disponible en la carpeta
    existing_files = sorted([int(f.split(".")[0]) for f in os.listdir(folder) if f.endswith(".png") and f.split(".")[0].isdigit()])
    next_number = (existing_files[-1] + 1) if existing_files else 1
    filename = os.path.join(folder, f"{next_number}.png")

    response = oauth.post(url, files=files)

    # Procesar la respuesta
    if response.status_code == 200:
        print("Solicitud exitosa")
        # Guardar la imagen resultante
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Imagen guardada como {filename}")
        # retornar el path de la imagen
        return filename
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print(response.text)
        return None

def test_model():
    """
    Ejecuta el modelo de cambio BAN sobre las imágenes de entrada y retorna la última imagen de resultado.
    """
    activate_command = (
        'cmd.exe /c "conda activate BAN && cd BAN && python test.py '
        'configs/ban/ban_vit-l14-clip_mit-b0_512x512_40k_levircd.py '
        'checkpoint/iter_40000.pth --show-dir resultados"'
    )
    result = subprocess.run(activate_command, shell=True)    
    
    if result.returncode != 0:
        print("Error al ejecutar el modelo:")
        print(result.stderr)
        return None

    result_dir = "BAN/resultados/vis_data/vis_image"
    if not os.path.exists(result_dir):
        print("No se encontró el directorio de resultados.")
        return None

    result_images = sorted(os.listdir(result_dir))
    if result_images:
        latest_result = os.path.join(result_dir, result_images[-1])
        print(f"Resultado generado: {latest_result}")
        try:
            result_image = Image.open(latest_result)
            return result_image 
        except Exception as e:
            print(f"No se pudo abrir la imagen: {e}")
            return latest_result
    else:
        print("No se encontraron imágenes de resultado.")
        return None    

def calcular_porcentaje_cambio(filename_mask):
    """
    Calcula el porcentaje de píxeles cambiados en una máscara binaria de cambio.
    """
    # Cargar imagen en escala de grises
    mascara = cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE)
    
    if mascara is None:
        print(f"⚠️ No se pudo cargar {filename_mask}")
        return

    # Contar píxeles con valor 255 (cambio) y el total de píxeles
    pixeles_cambio = np.count_nonzero(mascara == 255)
    total_pixeles = mascara.size  # Total de píxeles en la imagen

    # Calcular porcentaje de cambio
    porcentaje_cambio = (pixeles_cambio / total_pixeles) * 100

    # Mostrar resultado
    print(f"Imagen: {os.path.basename(filename_mask)} - Porcentaje de cambio: {porcentaje_cambio:.2f}%")
    return porcentaje_cambio

def porcentaje_por_indice(img1, img2):
    """
    Calcula el porcentaje de cambio entre clases de cobertura de suelo entre dos imágenes clasificadas.
    """
    RAMP = [
        ((62, 38, 21), "suelo_desnudo"),
        ((92, 58, 33), "suelo_poco_vegetado"),
        ((139, 90, 43), "suelo_moderadamente_vegetado"),
        ((166, 124, 82), "suelo_con_vegetacion_escasa"),
        ((133, 185, 110), "veg_palida"),
        ((92, 164, 80), "veg_media"),
        ((56, 140, 53), "veg_intensa"),
        ((39, 100, 25), "veg_densa"),
    ]
    AGUA = (0, 0, 255)
    CLASES = [c for _, c in RAMP] + ["agua"]

    # Agrupaciones
    SUELO = {"suelo_desnudo", "suelo_poco_vegetado", "suelo_moderadamente_vegetado", "suelo_con_vegetacion_escasa"}
    VEGETACION = {"veg_palida", "veg_media", "veg_intensa", "veg_densa"}
    AGUA_SET = {"agua"}

    def clasificar_imagen(img):
        arr = np.array(img)
        shape = arr.shape[:2]
        arr = arr.reshape(-1, 3)
        clases_idx = np.zeros(arr.shape[0], dtype=int)

        # Primero, asigna agua
        mask_agua = np.all(np.abs(arr - AGUA) <= 10, axis=1)
        clases_idx[mask_agua] = len(CLASES) - 1  # "agua"

        # Para los demás, busca el color de rampa más cercano
        ramp_colors = np.array([color for color, _ in RAMP])
        for idx in np.where(~mask_agua)[0]:
            dists = np.linalg.norm(ramp_colors - arr[idx], axis=1)
            clases_idx[idx] = np.argmin(dists)

        return clases_idx.reshape(shape)

    def clase_general(nombre):
        if nombre in SUELO:
            return "No Vegetación"
        elif nombre in VEGETACION:
            return "Vegetación"
        elif nombre in AGUA_SET:
            return "Agua"
        else:
            return "Otro"

    def analizar_cambios(img1, img2):
        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")
        assert img1.size == img2.size, "Las imágenes deben tener el mismo tamaño"

        clases1 = clasificar_imagen(img1)
        clases2 = clasificar_imagen(img2)

        total = clases1.size
        cambios = {}

        for i, c1 in enumerate(CLASES):
            for j, c2 in enumerate(CLASES):
                count = np.sum((clases1 == i) & (clases2 == j))
                if count > 0:
                    cambios[(c1, c2)] = count
        
        # Agrupar cambios
        resumen = {}
        for (c1, c2), count in cambios.items():
            g1 = clase_general(c1)
            g2 = clase_general(c2)
            key = (g1, g2)
            resumen[key] = resumen.get(key, 0) + count

        print(f"Total de píxeles analizados: {total}")
        print("Resumen de cambios agrupados (solo cambios):")
        for (g1, g2), count in resumen.items():
            if g1 != g2:
                print(f"{g1} -> {g2}: {count} ({count/total*100:.2f}%)")
        # Retornar el resumen de cambios (solo cambios)
        return {k: v for k, v in resumen.items() if k[0] != k[1]}, total

    porcentajes, total = analizar_cambios(img1, img2)
    return porcentajes, total

def generar_analisis_ollama3(ubicacion, cambios):
    """
    Genera un análisis textual de los cambios detectados usando un modelo de lenguaje local.
    """
    prompt = f"""
    Analiza los siguientes cambios bitemporales en tipos de suelo obtenidos de imágenes SAR para la ubicación: {ubicacion}. 
    Los datos de cambio son: {cambios}

    1. **Contexto Geográfico**:
    - Considerando que el área de estudio es {ubicacion['Dirección']} en {ubicacion['País']}, ¿cómo podrían influir las características típicas de esta región (clima, topografía, usos de suelo comunes) en los cambios observados?

    2. **Hallazgos Clave**:
    - Identifica los 3 cambios más significativos (mayores porcentajes) y su posible relación con actividades humanas o fenómenos naturales típicos de {ubicacion['Estado']}.
    - Ejemplo: "El cambio 'Vegetación → No Vegetación: 25.25%' podría relacionarse con [actividad agrícola, actividad ganadera] común en Morona Santiago".

    3. **Interpretación Ambiental**:
    - ¿Qué porcentajes son atípicos para un área de {ubicacion['Ciudad']}?
    - Destaca coincidencias con problemas ambientales conocidos de la región (ej: deforestación en Amazonía ecuatoriana).

    4. **Recomendaciones Específicas**:
    - 2 acciones prioritarias adaptadas a las políticas ambientales de {ubicacion['País']}.
    - Sugiere instituciones locales (ej: Ministerio del Ambiente de Ecuador) que deberían recibir estos resultados.

    Proporciona el análisis en español con terminología técnica pero accesible, usando comparaciones porcentuales cuando sea relevante.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}  # Para mayor precisión en datos técnicos
            }
        )

        print(response.json()["response"])
        return response.json()["response"]
    except:
        print("Error al comunicarse con el modelo de lenguaje.")
        return False