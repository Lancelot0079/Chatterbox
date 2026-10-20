# Texto a voz con Chatterbox TTS

Convierte texto en español a un audio `.wav` usando [Chatterbox Multilingual TTS](https://github.com/resemble-ai/chatterbox) con PyTorch. Funciona en CPU, y usa GPU NVIDIA (CUDA) o Mac (MPS) si están disponibles.

## Requisitos

- [uv](https://docs.astral.sh/uv/) y Python 3.12 o superior
- Conexión a internet la primera vez (descarga el modelo, varios GB)

## Ejecución

```bash
uv sync
uv run text-to-speech-chatterbox-tts
```

El programa pide un texto en español (con Enter usa uno de ejemplo) y guarda el resultado en `audio.wav` (24 kHz). En CPU un texto corto tarda alrededor de uno o dos minutos.

## Parámetros de generación

| Parámetro | Valor | Efecto |
|---|---|---|
| `language_id` | `"es"` | Idioma del texto |
| `exaggeration` | 0.7 | Expresividad o emoción de la voz |
| `cfg_weight` | 0.4 | Ritmo: valores bajos dan una voz más pausada |
| `temperature` | 0.8 | Variabilidad: más alta es menos predecible |
| `repetition_penalty` | 1.2 | Penaliza repeticiones extrañas |
| `min_p` | 0.05 | Descarta opciones muy improbables |
| `top_p` | 1.0 | Muestreo abierto |

## Usar la GPU NVIDIA

Por defecto `uv` instala PyTorch para CPU en Windows. Para usar CUDA hay que instalar torch con CUDA siguiendo la [guía de uv para PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/). El código detecta la GPU automáticamente.

## Estructura

```
├── pyproject.toml
├── uv.lock
├── src/text_to_speech_chatterbox_tts/__init__.py   # función main()
└── audio.wav                                       # audio de ejemplo
```
