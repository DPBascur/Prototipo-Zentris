import os
import wave
import tempfile
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import azure.cognitiveservices.speech as speechsdk
import json

# Control de volumen
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# === CONFIGURACIÓN ===
DISPOSITIVO_MIC = 1  # Cambia al índice de tu micrófono FANTECH
AZURE_KEY = "DyUlCWajpsyJdiI2wt6zyuwfAdD5bPw4jGgpi4q5fdZcBuWnzscFJQQJ99BGACYeBjFXJ3w3AAAYACOGwA2L"
AZURE_REGION = "eastus"

# === MODELO WHISPER ===
model = WhisperModel("medium", device="cuda", compute_type="float16")

# === HISTORIAL Y MEMORIA ===
chat_history = []
MEMORIA_PATH = "memoria.json"

def cargar_memoria():
    if os.path.exists(MEMORIA_PATH):
        with open(MEMORIA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def guardar_memoria(memoria):
    with open(MEMORIA_PATH, "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=2)

memoria = cargar_memoria()

# === SÍNTESIS DE VOZ CON AZURE ===
def say(text):
    print("🗣️ Zentris:", text)
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = "es-MX-JorgeNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("❌ Error al sintetizar voz:", result.reason)

# === GRABACIÓN DE AUDIO ===
def record_audio(fs=44100, silence_threshold=0.01, silence_duration=1.0, max_duration=30, device=None):
    print("🎙️ Escuchando...")
    audio_frames, silence_counter = [], 0
    chunk_size = int(0.1 * fs)
    max_chunks = int(max_duration / 0.1)
    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', device=device) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            audio_frames.append(chunk)
            rms = np.sqrt(np.mean(chunk**2))
            silence_counter = silence_counter + 0.1 if rms < silence_threshold else 0
            if silence_counter >= silence_duration:
                break
    audio = np.concatenate(audio_frames)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with wave.open(f.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        sf.write("debug_input.wav", audio, fs)
        return f.name

# === TRANSCRIPCIÓN ===
def transcribe(file):
    segments, _ = model.transcribe(file, beam_size=5, language='es')
    return " ".join([s.text for s in segments]).strip()

# === FUNCIONES DE MEMORIA ===
def procesar_memoria_usuario(texto):
    if texto.lower().startswith("recuerda que"):
        info = texto[12:].strip()
        if ":" in info:
            clave, valor = [i.strip() for i in info.split(":", 1)]
            memoria[clave.lower()] = valor
            guardar_memoria(memoria)
            return f"Lo recordaré, {clave} es {valor}."
        else:
            return "Por favor usa el formato: recuerda que [clave]: [valor]."
    elif texto.lower().startswith("¿qué recuerdas") or "qué recuerdas" in texto.lower():
        if not memoria:
            return "Aún no tengo recuerdos guardados."
        return "Esto es lo que recuerdo:\n" + "\n".join(f"- {k}: {v}" for k, v in memoria.items())
    return None

# === ABRIR PROGRAMAS ===
def ejecutar_comando_local(texto):
    texto_l = texto.lower()
    comandos = {
        "abrir bloc de notas": "notepad",
        "abrir calculadora": "calc",
        "abrir navegador": "start chrome",
        "abrir vscode": "code",
        "abrir explorador": "explorer",
        # Rutas típicas de Spotify en Windows
        "abrir spotify": [
            r'start "" "%USERPROFILE%\\AppData\\Roaming\\Spotify\\Spotify.exe"',
            r'start "" "%USERPROFILE%\\AppData\\Local\\Microsoft\\WindowsApps\\Spotify.exe"'
        ]
    }
    # Detección flexible para el teclado en pantalla
    if ("eleva el teclado" in texto_l) or ("teclado en pantalla" in texto_l) or ("abrir teclado" in texto_l):
        os.system("osk")
        return "Ejecutando teclado en pantalla."
    for clave, comando in comandos.items():
        if clave in texto_l:
            if isinstance(comando, list):
                for cmd in comando:
                    exit_code = os.system(cmd)
                    if exit_code == 0:
                        return f"Ejecutando {clave}."
                return "No se pudo encontrar Spotify en las rutas conocidas."
            else:
                os.system(comando)
                return f"Ejecutando {clave}."
    return None

# === CONTROL DE VOLUMEN ===
def controlar_volumen(texto):
    texto = texto.lower()
    dispositivos = AudioUtilities.GetSpeakers()
    interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
    current = volumen.GetMasterVolumeLevelScalar()

    if "sube el volumen" in texto:
        volumen.SetMasterVolumeLevelScalar(min(1.0, current + 0.1), None)
        return "Subiendo el volumen."
    elif "baja el volumen" in texto:
        volumen.SetMasterVolumeLevelScalar(max(0.0, current - 0.1), None)
        return "Bajando el volumen."
    elif "silencia" in texto or "mute" in texto:
        volumen.SetMute(1, None)
        return "Volumen silenciado."
    elif "reactiva el volumen" in texto or "quita el mute" in texto:
        volumen.SetMute(0, None)
        return "Volumen reactivado."
    return None

# === LLAMADA A OLLAMA ===
def ask_ollama(prompt):
    system_prompt = (
        "Tu nombre es Zentris, eres un asistente de inteligencia artificial como Jarvis.\n"
        "El usuario se llama Daniel. Dirígete a él por su nombre.\n"
        "Mantén una conversación natural, útil y profesional.\n"
        "Si el usuario te dice 'recuerda que', debes registrar esa información.\n"
        "Nunca olvides que eres Zentris, el asistente personal de Daniel.\n"
    )
    full_prompt = system_prompt + "\n"
    for q, a in chat_history[-5:]:
        full_prompt += f"Daniel: {q}\nZentris: {a}\n"
    full_prompt += f"Daniel: {prompt}\nZentris:"

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": full_prompt, "stream": False}
        )
        response = r.json()['response'].strip()
        chat_history.append((prompt, response))
        return response
    except Exception as e:
        print("❌ Error al consultar Ollama:", e)
        return "Lo siento, hubo un problema al generar mi respuesta."

# === BUCLE PRINCIPAL ===
if __name__ == "__main__":
    say("Hola, soy Zentris. ¿En qué te puedo ayudar?")

    while True:
        try:
            audio_file = record_audio(device=DISPOSITIVO_MIC)
            query = transcribe(audio_file)
            os.remove(audio_file)

            if not query.strip():
                continue

            print("🧠 Tú dijiste:", query)
            if "salir" in query.lower():
                say("Hasta luego.")
                break

            # 1. Memoria
            respuesta_memoria = procesar_memoria_usuario(query)
            if respuesta_memoria:
                say(respuesta_memoria)
                continue

            # 2. Comandos locales
            respuesta_local = ejecutar_comando_local(query)
            if respuesta_local:
                say(respuesta_local)
                continue

            # 3. Volumen
            respuesta_volumen = controlar_volumen(query)
            if respuesta_volumen:
                say(respuesta_volumen)
                continue

            # 4. IA
            print("🤔 Pensando...")
            respuesta = ask_ollama(query)
            say(respuesta)

        except KeyboardInterrupt:
            say("Interrumpido.")
            break
