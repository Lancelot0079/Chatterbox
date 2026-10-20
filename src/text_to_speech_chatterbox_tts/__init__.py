import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

TEXTO_POR_DEFECTO = (
    "Hola, bienvenidos a inteligencia artificial generativa. "
    "Vamos a crear audios, imágenes, texto y, si el PC aguanta, videos."
)
SALIDA = "audio.wav"


def elegir_dispositivo():
    # cuda - nvidia | mps - Mac con chip M | cpu - cualquier equipo
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    dispositivo = elegir_dispositivo()
    print(f"Cargando modelo en {dispositivo}...")
    modelo = ChatterboxMultilingualTTS.from_pretrained(device=dispositivo)

    texto = input("Escribe el texto en español (Enter para usar el de ejemplo): ").strip()
    if not texto:
        texto = TEXTO_POR_DEFECTO

    print("Generando audio...")
    audio = modelo.generate(
        texto,
        language_id="es",
        exaggeration=0.7,  # Expresión o emoción
        cfg_weight=0.4,  # Ritmo: más bajo = más pausado
        temperature=0.8,  # Variabilidad: más alta = menos predecible
        repetition_penalty=1.2,  # Penaliza repeticiones extrañas
        min_p=0.05,  # Filtra opciones improbables durante la generación
        top_p=1.0,  # Muestreo probabilístico, 1.0 lo deja abierto
    )

    ta.save(SALIDA, audio, modelo.sr)
    print(f"Audio guardado en {SALIDA}.")


if __name__ == "__main__":
    main()
