# Manual de instalación y guía de usuario

### MediSynth – Generador de Imágenes Médicas Sintéticas

Generador de imágenes tipo rayos X utilizando Stable Diffusion 1.5 + LoRA, diseñado exclusivamente con fines educativos e investigativos.
Permite generar radiografías sintéticas de tórax controladas mediante prompts y configuraciones avanzadas.

## Instalación
1. Clonar el repositorio
```bash
git clone https://github.com/jearx85/medisynth.git
cd medisynth
```

2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. Instalar dependencias

3.1 Instalar PyTorch (según tu GPU)

Ejemplo para CUDA 12.1:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Para CPU:
```bash
pip install torch
```
3.2 Instalar librerías del proyecto
```bash
pip install -r requirements.txt
```
## Configuración del Proyecto

El modelo base utilizado es:
```bash
runwayml/stable-diffusion-v1-5
```

El LoRA que ajusta el modelo:
```bash
jearx85/medisyn-lora
```

Ambos se descargan automáticamente desde HuggingFace la primera vez que se ejecuta la aplicación.

## Ejecutar la aplicación

Ejecuta:
```bash
python gradio_app.py
```

La aplicación abrirá una interfaz web de Gradio con una URL local y también se generará una URL pública temporal.


## Guía de Usuario

Al abrir la interfaz, verás:

1. Selección de tipo de caso

   - Normal

   - Neumonía leve

   - Neumonía moderada

   - Neumonía severa

2. Descripción adicional

Puedes agregar detalles como:

- opacity in right lung
- ground glass pattern
- mild cardiomegaly

3. Ajustes avanzados

Dentro del accordion:

* Steps: controla la calidad (recomendado: 40–70)

* Guidance scale: controla qué tanto el modelo sigue el prompt

Seed: pon -1 para aleatorio o un número fijo para reproducibilidad

4. Botón "Generar"

* Genera la imagen y muestra:

* la radiografía sintética

* el prompt construido

* la seed de generación
---
### Notas éticas y de uso

Las imágenes generadas son sintéticas, no reales.

No deben usarse para diagnóstico médico.

El dataset utilizado es público, anonimizado y no contiene información personal.

El propósito del proyecto es exclusivamente educación e investigación.

---
### Créditos

Modelo base: Stable Diffusion 1.5

Fine-tuning LoRA: MediSynth LoRA – by Jeisson Araque

Interfaz: Gradio Blocks

Librerías: Diffusers, PEFT, Transformers