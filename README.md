# Detección de Cambios en Imágenes SAR Bitemporales

Este proyecto permite detectar y analizar cambios en imágenes satelitales de radar de apertura sintética (SAR) usando imágenes bitemporales Sentinel-1. La arquitectura de detección de cambios está basada en la red [BAN](https://arxiv.org/abs/2312.01163), implementada sobre [Open-CD](https://github.com/likyoo/open-cd). Incluye una interfaz web interactiva desarrollada en Streamlit, integración con modelos de deep learning y la opción de interpretación automática mediante Ollama3.

---

## Descripción

La aplicación descarga imágenes SAR desde Sentinel Hub, genera máscaras de cambio mediante modelos entrenados y permite interpretar los resultados con un modelo de lenguaje (Ollama3). 

Además de mostrar la máscara de cambio, la aplicación calcula y presenta los porcentajes de cambio entre diferentes coberturas (como vegetación, suelo, agua, etc.), facilitando así el análisis cuantitativo de los resultados.

Este proyecto es parte de un trabajo de titulación de la Universidad de Cuenca. El paper asociado está en proceso de publicación.

---

## Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/mateocalle7/Deteccion-de-Cambios-SAR.git
cd tu_repositorio
```

### 2. Crear entorno Conda y activar

```bash
conda create -n BAN python=3.8 -y
conda activate BAN
```

### 3. Instalar PyTorch y torchvision

Por ejemplo, para instalar torch==2.0.0 con CUDA==11.8:

```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### 4. Instalar toolkits de OpenMMLab

```bash
pip install -U openmim
mim install mmengine==0.10.1
mim install mmcv==2.1.0
mim install mmpretrain==1.1.1
pip install mmsegmentation==1.2.2
pip install mmdet==3.2.0
```

### 5. Instalar dependencias del proyecto

```bash
pip install -r requirements.txt
```

### 6. Instalar Ollama y el modelo Ollama3 (opcional, solo si se desea interpretación automática)

- Descarga e instala Ollama desde [https://ollama.com/download](https://ollama.com/download)
- Instala el modelo Ollama3:
  ```bash
  ollama pull ollama3
  ```

### 7. Obtener credenciales de Sentinel Hub

- Regístrate y genera tus credenciales en [Sentinel Hub](https://www.sentinel-hub.com/)
- Copia tu `client_id` y `client_secret` en el archivo `config.py`:
  ```python
  client_id = "TU_CLIENT_ID"
  client_secret = "TU_CLIENT_SECRET"
  ```

### 8. Descargar el modelo entrenado

Descarga el modelo [iter_40000.pth](https://huggingface.co/datasets/mateocalle7/Modelo-Deteccion-Cambios-SAR/tree/main) y colócalo en la carpeta `checkpoint/` del proyecto.

---

## Ejecución

Para iniciar la aplicación web localmente, ejecuta:

```bash
streamlit run app.py
```

Se abrirá una interfaz web donde solo debes ingresar las coordenadas y las fechas bitemporales de interés.

---

## Créditos

- Basado en la red [BAN](https://github.com/likyoo/BAN) y [Open-CD](https://github.com/likyoo/open-cd).
- Autores: Mateo Sebastián Calle Siavichay, Santiago Ismael Tigre Cajas
- Universidad de Cuenca


## Contacto

Para dudas o contribuciones, puedes escribir a:

- mateo.calle@ucuenca.edu.ec
- santiago.tigre@ucuenca.edu.ec
- o dejar un issue en el repositorio de GitHub.