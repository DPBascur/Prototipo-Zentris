# Prototipo-Zentris

Asistente personal en Telegram con integración local a Ollama, memoria conversacional y análisis de imágenes.

El bot está orientado a uso personal, con restricción por chat_id para evitar accesos no autorizados.

## Características

- Chat en Telegram con respuestas en español.
- Enrutado dinámico de modelo según complejidad del mensaje.
- Streaming de respuesta para mejorar experiencia de uso.
- Memoria corta persistente en archivo JSON.
- Memoria semántica con ChromaDB y embeddings.
- Extracción automática de perfil del usuario.
- Análisis de imágenes con modelo multimodal de Ollama.
- Traducción automática a español cuando el modelo responde en otro idioma.
- Arquitectura modular para facilitar mantenimiento y escalabilidad.

## Arquitectura

El proyecto está organizado por capas:

- app/config.py
	Carga y validación de configuración.

- app/bootstrap.py
	Composición de dependencias y creación de la aplicación de Telegram.

- app/application/
	Lógica de negocio principal del asistente.

- app/infrastructure/
	Integraciones concretas: Ollama, almacenamiento, traducción y memoria vectorial.

- app/presentation/
	Handlers de Telegram para texto e imágenes.

- bot.py
	Punto de entrada para ejecutar el bot.

## Requisitos

- Python 3.10 o superior.
- Ollama instalado y en ejecución.
- Un bot de Telegram creado con BotFather.

## Instalación

1. Crear y activar entorno virtual

En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Instalar dependencias

```powershell
pip install python-telegram-bot requests httpx python-dotenv chromadb sentence-transformers langdetect deep-translator
```

## Configuración

Crea un archivo .env en la raíz del proyecto con este contenido base:

```env
TELEGRAM_TOKEN=TU_TOKEN_DE_BOTFATHER
TELEGRAM_ALLOWED_CHAT_ID=TU_CHAT_ID

OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_TAGS_URL=http://localhost:11434/api/tags

FAST_MODEL=phi3
BALANCED_MODEL=mistral
REASONING_MODEL=mistral
VISION_MODEL=llava

APP_MODE=telegram

MAX_HISTORY=1200
REQUEST_TIMEOUT=60
STREAM_UPDATE_INTERVAL=0.9
MIN_EDIT_DELTA=20

VOICE_RECORD_SECONDS=5
VOICE_SAMPLE_RATE=16000
VOICE_MODEL_SIZE=base
VOICE_LANGUAGE=es
VOICE_INPUT_DEVICE=
VOICE_TTS_ENABLED=true
VOICE_TTS_RATE=170
VOICE_TTS_PROVIDER=local

AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_VOICE=es-MX-JorgeNeural

VOICE_NOISE_CALIBRATION_SECONDS=1.0
VOICE_NOISE_GATE_MULTIPLIER=2.0
VOICE_MIN_SPEECH_SECONDS=0.5
VOICE_SILENCE_TIMEOUT_SECONDS=1.0
VOICE_MAX_RECORD_SECONDS=8.0

VOICE_STT_BEAM_SIZE=6
VOICE_STT_BEST_OF=4
VOICE_STT_TEMPERATURE=0.0
VOICE_STT_NO_SPEECH_THRESHOLD=0.6
VOICE_STT_LOGPROB_THRESHOLD=-1.2
VOICE_VAD_MIN_SILENCE_MS=500

VOICE_FORCE_FAST_MODEL=true
VOICE_PROFILE_EXTRACT_LLM=false

FORCE_GPU=false
STT_DEVICE=cpu
EMBEDDING_DEVICE=cpu
```

Modos disponibles en APP_MODE:

- telegram: solo bot de Telegram.
- local: solo asistente local por voz.
- hybrid: Telegram y voz local al mismo tiempo.

Configuracion de GPU:

- `FORCE_GPU=true` fuerza CUDA para STT y embeddings, y valida CUDA al iniciar.
- Si `FORCE_GPU=false`, puedes elegir manualmente:
	- `STT_DEVICE=cpu|cuda`
	- `EMBEDDING_DEVICE=cpu|cuda`

## Modelos en Ollama

Debes tener instalados los modelos que declares en .env.

Ejemplo:

```powershell
ollama pull phi3
ollama pull mistral
ollama pull llava
```

Verificar modelos instalados:

```powershell
curl.exe -s http://localhost:11434/api/tags
```

## Ejecución

Desde la raíz del proyecto:

```powershell
python bot.py
```

Para modo local por voz, instala también:

```powershell
pip install numpy sounddevice soundfile faster-whisper pyttsx3
```

Listar micrófonos disponibles y su índice:

```powershell
python scripts/list_audio_devices.py
```

Luego copia ese índice en `VOICE_INPUT_DEVICE` dentro de `.env`.

Para usar voz de Azure en modo local:

1. Instala el SDK:

```powershell
pip install azure-cognitiveservices-speech
```

2. Configura en `.env`:

```env
VOICE_TTS_PROVIDER=azure
AZURE_SPEECH_KEY=tu_clave
AZURE_SPEECH_REGION=tu_region
AZURE_SPEECH_VOICE=es-MX-JorgeNeural
```

Si todo está correcto, verás en consola:

```text
Zentris optimizado corriendo...
Application started
```

## Uso

- Envía mensajes de texto al bot para conversación normal.
- Envía una imagen con o sin descripción para análisis visual.
- Si agregas caption en la foto, el bot intentará responder exactamente a esa instrucción.

## Seguridad

- El bot responde solo a un chat_id permitido.
- Se ignoran chats no autorizados y grupos.

Recomendaciones:

- No subas .env al repositorio.
- Si expones el token, revócalo en BotFather y genera uno nuevo.

## Archivos de datos

- memory.json
	Historial corto por usuario.

- profile.json
	Perfil extraído automáticamente del usuario.

## Problemas comunes

1. El bot no responde en Telegram

- Verifica TELEGRAM_ALLOWED_CHAT_ID.
- Asegúrate de hablarle por chat privado.
- Revisa que aparezca Application started en consola.

2. Error al analizar imágenes

- Verifica que VISION_MODEL exista en Ollama.
- Instala el modelo con ollama pull llava.

3. Respuestas lentas al inicio

- Es normal durante la primera carga de sentence-transformers.
- Las siguientes ejecuciones suelen ser más rápidas.

4. Error con FORCE_GPU=true

- Verifica que CUDA este disponible y los drivers instalados.
- Si no tienes GPU compatible, usa `FORCE_GPU=false`.

## Estado actual

El proyecto se encuentra en una base funcional y modular, lista para seguir creciendo con:

- comandos de Telegram,
- tests automáticos,
- métricas de latencia,
- despliegue continuo.
