import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
model_base = "runwayml/stable-diffusion-v1-5"
lora_path = "jearx85/medisyn-lora"

print(f"Usando dispositivo: {device}")
print(f"Usando dtype: {dtype}")
print("Cargando UNet base...")

# ============================================================
# CARGAR MODELO BASE
# ============================================================

unet = UNet2DConditionModel.from_pretrained(
    model_base,
    subfolder="unet",
    torch_dtype=dtype
).to(device)

print("Creando pipeline...")
pipeline = StableDiffusionPipeline.from_pretrained(
    model_base,
    unet=unet,
    torch_dtype=dtype,
    safety_checker=None
).to(device)

# OPTIMIZACIONES
pipeline.enable_attention_slicing(slice_size=1)  # Reduce uso de memoria

if device == "cpu":
    print("Ejecutando en CPU - La generación tomará 2-5 minutos")
    # En CPU solo usamos attention slicing, no cpu_offload
else:
    # Solo en GPU: mueve capas entre CPU y GPU
    pipeline.enable_model_cpu_offload()

# ============================================================
# CARGAR LoRA
# ============================================================

print("Cargando LoRA desde HuggingFace:", lora_path)

try:
    from peft import PeftModel, LoraConfig
    
    # Método 1: Intentar cargar con PeftModel (formato PEFT)
    try:
        pipeline.unet = PeftModel.from_pretrained(
            pipeline.unet,
            lora_path,
            adapter_name="medisyn"
        )
        print("LoRA cargado correctamente con PeftModel")
    except Exception as e1:
        print(f"No se pudo cargar con PeftModel: {e1}")
        
        # Método 2: Intentar con load_lora_weights (formato Diffusers nativo)
        try:
            pipeline.load_lora_weights(lora_path)
            print("LoRA cargado correctamente con load_lora_weights")
        except Exception as e2:
            print(f"No se pudo cargar con load_lora_weights: {e2}")
            print("Continuando sin LoRA...")
            
except Exception as e:
    print(f"Error general cargando LoRA: {e}")
    print("Continuando sin LoRA...")

# ============================================================
# FUNCIÓN DE GENERACIÓN
# ============================================================

def generate_xray(disease_type, custom_text, steps, guidance, seed):

    base_prompts = {
        "Normal": "medical normal healthy chest x-ray, clear lungs, no infection",
        "Neumonía Leve": "medical chest x-ray showing mild pneumonia, slight lung infection",
        "Neumonía Moderada": "medical chest x-ray showing pneumonia, lung infection, opacity in lungs",
        "Neumonía Severa": "medical chest x-ray showing severe pneumonia, bilateral lung infection",
    }

    prompt = base_prompts[disease_type]

    if custom_text:
        prompt += f", {custom_text}"

    prompt += ", high quality radiograph, diagnostic quality"

    if seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device).manual_seed(int(seed))

    # OPTIMIZACIÓN: Reduce pasos en CPU
    if device == "cpu" and steps > 30:
        steps = 30
        print(f"Reduciendo a {steps} pasos para CPU")

    try:
        image = pipeline(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator
        ).images[0]

        info = f"**Prompt usado:** {prompt}\n\n**Seed:** {seed}\n\n**Pasos:** {steps}"

        return image, info

    except Exception as e:
        return None, f"Error al generar: {str(e)}"


# ============================================================
# INTERFAZ DE GRADIO
# ============================================================

with gr.Blocks(title="MediSynth - Generador Médico") as demo:

    gr.Markdown(f"""
    # MediSynth - Generador de Imágenes Médicas

    Genera rayos X sintéticos usando **Stable Diffusion + LoRA** entrenado en imágenes médicas reales.

    **Uso académico — NO es apto para diagnóstico clínico.**
    
    {'GPU detectada - Generación rápida' if device == 'cuda' else 'CPU detectada - Generación lenta (2-5 min)'}
    """)

    with gr.Row():
        with gr.Column():
            disease_type = gr.Dropdown(
                choices=["Normal", "Neumonía Leve", "Neumonía Moderada", "Neumonía Severa"],
                value="Neumonía Moderada",
                label="Tipo de Imagen"
            )

            custom_text = gr.Textbox(
                label="Descripción Adicional (Opcional)",
                placeholder="Ej: bilateral, lower lobe, diffuse...",
                lines=2
            )

            with gr.Accordion("Configuración Avanzada", open=False):
                # Valores por defecto optimizados para CPU
                default_steps = 25 if device == "cpu" else 50
                steps = gr.Slider(
                    10, 50 if device == "cpu" else 100, 
                    value=default_steps, 
                    step=5, 
                    label=f"Pasos (recomendado: {default_steps})"
                )
                guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance Scale")
                seed = gr.Number(value=-1, label="Semilla (-1 aleatoria)", precision=0)

            generate_btn = gr.Button("Generar Imagen", variant="primary")
            
            if device == "cpu":
                gr.Markdown("*Tiempo estimado: 2-5 minutos*")

        with gr.Column():
            output_image = gr.Image(label="Imagen Generada", type="pil")
            output_info = gr.Markdown()

    generate_btn.click(
        fn=generate_xray,
        inputs=[disease_type, custom_text, steps, guidance, seed],
        outputs=[output_image, output_info]
    )

demo.launch()