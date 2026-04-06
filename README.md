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

MAX_HISTORY=1200
REQUEST_TIMEOUT=60
STREAM_UPDATE_INTERVAL=0.9
MIN_EDIT_DELTA=20
```

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

## Estado actual

El proyecto se encuentra en una base funcional y modular, lista para seguir creciendo con:

- comandos de Telegram,
- tests automáticos,
- métricas de latencia,
- despliegue continuo.
