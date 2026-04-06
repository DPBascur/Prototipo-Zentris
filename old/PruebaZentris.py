import time
import threading
import serial
import os
import wave
import tempfile
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import json
from faster_whisper import WhisperModel
from pocketsphinx import LiveSpeech
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

# === CONFIGURACIÓN ===
strWakeWord         = "oye zentris"
fltWakeThreshold    = 1e-20
objWhisperModel     = WhisperModel("medium", device="cuda", compute_type="float16")
intMicDeviceIndex   = 1  # Cambiar al índice del micrófono
strMemoriaPath      = "memoria.json"
strAzureKey         = "DyUlCWajpsyJdiI2wt6zyuwfAdD5bPw4jGgpi4q5fdZcBuWnzscFJQQJ99BGACYeBjFXJ3w3AAAYACOGwA2L"
strAzureRegion      = "eastus"

# === SERIAL PARA ARDUINO ===
objSerialArduino = None
try:
    objSerialArduino = serial.Serial("COM3", 9600, timeout=1)
except:
    print("⚠️ Arduino no conectado.")

# === CARGA MEMORIA ===
dictMemoria = {}
if os.path.exists(strMemoriaPath):
    with open(strMemoriaPath, "r", encoding="utf-8") as f:
        dictMemoria = json.load(f)

def guardar_memoria():
    with open(strMemoriaPath, "w", encoding="utf-8") as f:
        json.dump(dictMemoria, f, ensure_ascii=False, indent=2)

# === TEXTO A VOZ CON AZURE ===
def decir_texto(strTexto):
    print("🗣️ Zentris:", strTexto)
    try:
        import azure.cognitiveservices.speech as speechsdk
        speech_config = speechsdk.SpeechConfig(subscription=strAzureKey, region=strAzureRegion)
        speech_config.speech_synthesis_voice_name = "es-MX-JorgeNeural"
        audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
        result = synthesizer.speak_text_async(strTexto).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("❌ Error TTS:", result.reason)
    except Exception as e:
        print("⚠️ Error Azure TTS:", e)

# === GRABACIÓN DE AUDIO ===
def grabar_audio():
    intFS = 16000
    intDuracion = 5
    arrAudio = sd.rec(intDuracion * intFS, samplerate=intFS, channels=1, dtype='int16', device=intMicDeviceIndex)
    sd.wait()
    strNombreArchivo = "comando_temp.wav"
    sf.write(strNombreArchivo, arrAudio, intFS)
    return strNombreArchivo

# === TRANSCRIPCIÓN ===
def transcribir_audio(strArchivo):
    segments, _ = objWhisperModel.transcribe(strArchivo, beam_size=5, language='es')
    strTexto = " ".join([seg.text for seg in segments])
    return strTexto.replace("centris", "Zentris").strip()

# === PROCESAMIENTO DE MEMORIA ===
def procesar_memoria(strTexto):
    strTextoLower = strTexto.lower()
    if strTextoLower.startswith("recuerda que"):
        strInfo = strTexto[12:].strip()
        if ":" in strInfo:
            strClave, strValor = [i.strip() for i in strInfo.split(":", 1)]
            dictMemoria[strClave.lower()] = strValor
            guardar_memoria()
            return f"Lo recordaré, {strClave} es {strValor}."
        else:
            return "Por favor usa el formato: recuerda que [clave]: [valor]."
    elif "qué recuerdas" in strTextoLower:
        if not dictMemoria:
            return "Aún no tengo recuerdos guardados."
        return "Esto es lo que recuerdo:\n" + "\n".join(f"- {k}: {v}" for k, v in dictMemoria.items())
    return None

# === COMANDOS LOCALES ===
def ejecutar_comando_local(strTexto):
    strTextoL = strTexto.lower()
    dictComandos = {
        "abrir bloc de notas": "notepad",
        "abrir calculadora": "calc",
        "abrir navegador": "start chrome",
        "abrir vscode": "code",
        "abrir explorador": "explorer"
    }
    for strClave, strComando in dictComandos.items():
        if strClave in strTextoL:
            os.system(strComando)
            return f"Ejecutando {strClave}."
    return None

# === VOLUMEN ===
def controlar_volumen(strTexto):
    dispositivos = AudioUtilities.GetSpeakers()
    interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
    actual = volumen.GetMasterVolumeLevelScalar()
    if "sube el volumen" in strTexto:
        volumen.SetMasterVolumeLevelScalar(min(1.0, actual + 0.1), None)
        return "Subiendo el volumen."
    elif "baja el volumen" in strTexto:
        volumen.SetMasterVolumeLevelScalar(max(0.0, actual - 0.1), None)
        return "Bajando el volumen."
    elif "silencia" in strTexto:
        volumen.SetMute(1, None)
        return "Volumen silenciado."
    elif "reactiva el volumen" in strTexto:
        volumen.SetMute(0, None)
        return "Volumen reactivado."
    return None

# === CONTROL DE ARDUINO ===
def controlar_dispositivo(strTexto):
    if objSerialArduino:
        if "enciende la luz" in strTexto:
            objSerialArduino.write(b'H')
            return "Encendiendo la luz."
        elif "apaga la luz" in strTexto:
            objSerialArduino.write(b'L')
            return "Apagando la luz."
    return None

# === LLAMADA A OLLAMA ===
def preguntar_ollama(strPrompt):
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi3", "prompt": strPrompt, "stream": False}
        )
        return r.json()['response'].strip()
    except Exception as e:
        return "Lo siento, hubo un error al generar la respuesta."

import os
from pocketsphinx import LiveSpeech

def detectar_wake_word():
    global bWakeWordDetectado
    strRutaModelo = os.path.abspath("model-es")  # convierte a ruta absoluta
    print("🔍 Cargando modelo:", strRutaModelo)

    speech = LiveSpeech(
        lm=False,
        keyphrase="oye zentris",
        kws_threshold=1e-20,
        dic="diccionario.dic",
        hmm=strRutaModelo
    )

    for frase in speech:
        print("🔔 Wake word detectada!")
        bWakeWordDetectado = True
        break





# === BUCLE PRINCIPAL ===
if __name__ == "__main__":
    decir_texto("Hola, soy Zentris. Espero tu comando cuando digas 'Oye Zentris'.")
    bWakeWordDetectado = False
    threading.Thread(target=detectar_wake_word, daemon=True).start()

    while True:
        if not bWakeWordDetectado:
            time.sleep(0.1)
            continue

        bWakeWordDetectado = False
        strArchivo = grabar_audio()
        strTextoUsuario = transcribir_audio(strArchivo)
        os.remove(strArchivo)

        if not strTextoUsuario.strip():
            continue

        print("🧠 Tú dijiste:", strTextoUsuario)

        for funcion in [procesar_memoria, ejecutar_comando_local, controlar_volumen, controlar_dispositivo]:
            strResp = funcion(strTextoUsuario)
            if strResp:
                decir_texto(strResp)
                break
        else:
            decir_texto(preguntar_ollama(strTextoUsuario))

        threading.Thread(target=detectar_wake_word, daemon=True).start()
