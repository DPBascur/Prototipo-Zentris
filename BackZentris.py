import numpy as np
import tempfile
import requests
import os
import wave
import json
import sounddevice as sd
import soundfile as sf
import azure.cognitiveservices.speech as speechsdk
from faster_whisper import WhisperModel

# === CONFIGURACIÓN ===
DISPOSITIVO_MIC = 1
AZURE_KEY = "DyUlCWajpsyJdiI2wt6zyuwfAdD5bPw4jGgpi4q5fdZcBuWnzscFJQQJ99BGACYeBjFXJ3w3AAAYACOGwA2L"
AZURE_REGION = "eastus"

# === CARGAR MEMORIA ===
MEMORIA_PATH = "memoria.json"

def cargar_memoria():
    if os.path.exists(MEMORIA_PATH):
        with open(MEMORIA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"conocimientos": {}}

def guardar_memoria(memoria):
    with open(MEMORIA_PATH, "w", encoding="utf-8") as f:
        json.dump(memoria, f, indent=2, ensure_ascii=False)

memoria = cargar_memoria()
nombre_usuario = memoria.get("nombre_usuario", "usuario")
nombre_asistente = memoria.get("nombre_asistente", "asistente")
descripcion_asistente = memoria.get("descripcion_asistente", "")
conocimientos = memoria.get("conocimientos", {})

# === MODELO WHISPER ===
model = WhisperModel("medium", device="cuda", compute_type="float16")

# === VOZ CON AZURE ===
def say(text):
    print(f"🗣️ {nombre_asistente}:", text)

    speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = "es-MX-JorgeNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    result = synthesizer.speak_text_async(text).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("❌ Error al sintetizar voz:", result.reason)

# === AUDIO ===
def record_audio(fs=44100, silence_threshold=0.01, silence_duration=1.0, max_duration=30, device=None):
    print("🎙️ Escuchando...")
    audio_frames = []
    silence_counter = 0
    chunk_size = int(0.1 * fs)
    max_chunks = int(max_duration / 0.1)

    with sd.InputStream(samplerate=fs, channels=1, dtype='float32', device=device) as stream:
        for _ in range(max_chunks):
            chunk, _ = stream.read(chunk_size)
            audio_frames.append(chunk)
            rms = np.sqrt(np.mean(chunk**2))
            if rms < silence_threshold:
                silence_counter += 0.1
            else:
                silence_counter = 0
            if silence_counter >= silence_duration:
                break

    audio = np.concatenate(audio_frames, axis=0)
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
    text = " ".join([segment.text for segment in segments])
    return text.strip()

# === PROCESAR "RECUERDA QUE..." ===
def procesar_memoria_usuario(texto):
    if texto.lower().startswith("recuerda que"):
        info = texto[len("recuerda que"):].strip()
        if ":" in info:
            clave, valor = info.split(":", 1)
        elif " es " in info:
            clave, valor = info.split(" es ", 1)
        else:
            clave = "dato"
            valor = info
        clave = clave.strip().capitalize()
        conocimientos[clave] = valor.strip()
        memoria["conocimientos"] = conocimientos
        guardar_memoria(memoria)
        return f"Entendido, recordaré que {clave.lower()} es {valor.strip()}."
    return None

# === CONVERSACIÓN ===
chat_history = []

def ask_ollama(prompt):
    full_prompt = descripcion_asistente + "\n"
    full_prompt += f"Conocimientos del usuario:\n"
    for k, v in conocimientos.items():
        full_prompt += f"- {k}: {v}\n"

    for q, a in chat_history[-5:]:
        full_prompt += f"{nombre_usuario}: {q}\n{nombre_asistente}: {a}\n"
    full_prompt += f"{nombre_usuario}: {prompt}\n{nombre_asistente}:"

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": full_prompt, "stream": False}
        ).json()['response'].strip()

        chat_history.append((prompt, response))
        return response

    except Exception as e:
        print("❌ Error al consultar Ollama:", e)
        return "Lo siento, hubo un problema al generar mi respuesta."
    

# === LOOP PRINCIPAL ===
if __name__ == "__main__":
    say(f"Hola {nombre_usuario}, soy {nombre_asistente}. ¿En qué te puedo ayudar hoy?")

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

            # Procesar si es memoria
            respuesta_memoria = procesar_memoria_usuario(query)
            if respuesta_memoria:
                say(respuesta_memoria)
                continue

            print("🤔 Pensando...")
            respuesta = ask_ollama(query)
            say(respuesta)

        except KeyboardInterrupt:
            say("Interrumpido.")
            break
