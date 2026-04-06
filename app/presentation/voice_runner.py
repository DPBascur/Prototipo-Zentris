import asyncio
import os
import re
import tempfile
import threading
import time

from app.infrastructure.language import enforce_spanish


class LocalVoiceRunner:
    def __init__(self, settings, logger, core, ollama_client):
        self.settings = settings
        self.logger = logger
        self.core = core
        self.ollama = ollama_client

        missing = []
        try:
            import numpy as np
            self.np = np
        except Exception:
            missing.append("numpy")

        try:
            import sounddevice as sd
            self.sd = sd
        except Exception:
            missing.append("sounddevice")

        try:
            import soundfile as sf
            self.sf = sf
        except Exception:
            missing.append("soundfile")

        try:
            from faster_whisper import WhisperModel
            self.WhisperModel = WhisperModel
        except Exception:
            missing.append("faster-whisper")

        self.tts = None
        self.azure_speechsdk = None
        self.noise_floor = 0.01
        self.interrupt_event = threading.Event()
        self.stop_listener_event = threading.Event()
        self._keyboard_listener_thread = None
        self._last_interrupt_at = 0.0

        self.msvcrt = None
        if os.name == "nt":
            try:
                import msvcrt as _msvcrt
                self.msvcrt = _msvcrt
            except Exception:
                self.msvcrt = None

        if settings.voice_tts_enabled and settings.voice_tts_provider == "azure":
            try:
                import azure.cognitiveservices.speech as speechsdk
                self.azure_speechsdk = speechsdk
                if not settings.azure_speech_key or not settings.azure_speech_region:
                    self.logger.warning(
                        "VOICE_TTS_PROVIDER=azure pero faltan AZURE_SPEECH_KEY/AZURE_SPEECH_REGION. Se usa fallback local."
                    )
                    self.azure_speechsdk = None
            except Exception as exc:
                self.logger.warning("No se pudo inicializar Azure Speech SDK: %s", exc)

        if settings.voice_tts_enabled and (settings.voice_tts_provider == "local" or self.azure_speechsdk is None):
            try:
                import pyttsx3
                self.tts = pyttsx3.init()
                self.tts.setProperty("rate", settings.voice_tts_rate)
            except Exception as exc:
                self.logger.warning("No se pudo inicializar TTS local (pyttsx3): %s", exc)

        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(
                "Faltan dependencias para modo local de voz: "
                f"{joined}. Instala con: pip install {joined}"
            )

        stt_compute_type = "float16" if settings.stt_device == "cuda" else "int8"
        self.stt_model = self.WhisperModel(
            settings.voice_model_size,
            device=settings.stt_device,
            compute_type=stt_compute_type,
        )
        self.calibrate_noise_floor()

    def _start_keyboard_listener(self) -> None:
        if self._keyboard_listener_thread and self._keyboard_listener_thread.is_alive():
            return
        if self.msvcrt is None:
            return

        self.stop_listener_event.clear()
        self._keyboard_listener_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
        self._keyboard_listener_thread.start()

    def _stop_keyboard_listener(self) -> None:
        self.stop_listener_event.set()

    def _keyboard_listener_loop(self) -> None:
        while not self.stop_listener_event.is_set():
            try:
                if self.msvcrt and self.msvcrt.kbhit():
                    while self.msvcrt.kbhit():
                        self.msvcrt.getch()

                    now = time.monotonic()
                    if now - self._last_interrupt_at < 0.35:
                        time.sleep(0.05)
                        continue

                    already_set = self.interrupt_event.is_set()
                    self.interrupt_event.set()
                    self._last_interrupt_at = now
                    try:
                        if self.tts:
                            self.tts.stop()
                    except Exception:
                        pass
                    if not already_set:
                        print("[Interrupcion] Turno cancelado por teclado.")
                time.sleep(0.05)
            except Exception:
                time.sleep(0.1)

    def _split_tts_chunks(self, text: str, max_words: int = 10) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        sentence_parts = re.split(r"(?<=[\.!\?])\s+", text)
        chunks = []
        for sentence in sentence_parts:
            sentence = sentence.strip()
            if not sentence:
                continue
            words = sentence.split()
            if len(words) <= max_words:
                chunks.append(sentence)
                continue
            for idx in range(0, len(words), max_words):
                chunk = " ".join(words[idx : idx + max_words]).strip()
                if chunk:
                    chunks.append(chunk)
        return chunks

    def normalize_transcription(self, text: str) -> str:
        normalized = f" {text.strip()} "
        replacements = {
            " centrist ": " zentris ",
            " centris ": " zentris ",
            " centriz ": " zentris ",
            " sentris ": " zentris ",
            " en tres ": " zentris ",
            " zen tres ": " zentris ",
            " vuelas en tres ": " hola zentris ",
            " buenas en tres ": " hola zentris ",
            " por la zentris ": " hola zentris ",
            " por la centrist ": " hola zentris ",
        }
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
            normalized = normalized.replace(src.capitalize(), dst.capitalize())
        return normalized.strip()

    def calibrate_noise_floor(self) -> None:
        fs = self.settings.voice_sample_rate
        duration = max(self.settings.voice_noise_calibration_seconds, 0.3)
        frames = int(duration * fs)
        try:
            audio = self.sd.rec(
                frames,
                samplerate=fs,
                channels=1,
                dtype="float32",
                device=self.settings.voice_input_device,
            )
            self.sd.wait()
            mono = audio[:, 0]
            rms = float(self.np.sqrt(self.np.mean(self.np.square(mono))))
            self.noise_floor = max(rms, 0.003)
            self.logger.info("Ruido base calibrado: %.5f", self.noise_floor)
        except Exception as exc:
            self.logger.warning("No se pudo calibrar ruido base: %s", exc)

    def _rms(self, chunk) -> float:
        return float(self.np.sqrt(self.np.mean(self.np.square(chunk))))

    def _apply_basic_noise_suppression(self, audio):
        mono = audio[:, 0]
        mono = mono - float(self.np.mean(mono))
        gate = max(self.noise_floor * self.settings.voice_noise_gate_multiplier, 0.004)
        # Atenua ruido bajo en lugar de eliminarlo para evitar perder fonemas suaves.
        mono = self.np.where(self.np.abs(mono) < gate, mono * 0.2, mono)
        return mono.reshape(-1, 1)

    def speak(self, text: str) -> None:
        print(f"Zentris: {text}")

        if self.settings.voice_tts_enabled and self.azure_speechsdk and self.settings.voice_tts_provider == "azure":
            try:
                speech_config = self.azure_speechsdk.SpeechConfig(
                    subscription=self.settings.azure_speech_key,
                    region=self.settings.azure_speech_region,
                )
                speech_config.speech_synthesis_voice_name = self.settings.azure_speech_voice
                audio_config = self.azure_speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
                synthesizer = self.azure_speechsdk.SpeechSynthesizer(
                    speech_config=speech_config,
                    audio_config=audio_config,
                )
                if self.interrupt_event.is_set():
                    return
                result = synthesizer.speak_text_async(text).get()
                if result.reason == self.azure_speechsdk.ResultReason.SynthesizingAudioCompleted:
                    return
                if result.reason == self.azure_speechsdk.ResultReason.Canceled:
                    details = self.azure_speechsdk.CancellationDetails(result)
                    self.logger.warning(
                        "Azure TTS cancelado (%s). Detalle: %s. Se usa fallback local.",
                        details.reason,
                        details.error_details,
                    )
                else:
                    self.logger.warning("Azure TTS no completado (%s), se usa fallback local", result.reason)
            except Exception as exc:
                self.logger.warning("Error en Azure TTS, se usa fallback local: %s", exc)

        if self.tts:
            try:
                for chunk in self._split_tts_chunks(text):
                    if self.interrupt_event.is_set():
                        try:
                            self.tts.stop()
                        except Exception:
                            pass
                        break
                    self.tts.say(chunk)
                    self.tts.runAndWait()
            except Exception as exc:
                self.logger.warning("Error en TTS local: %s", exc)

    def record_audio_file(self) -> str | None:
        fs = self.settings.voice_sample_rate
        chunk_seconds = 0.1
        chunk_frames = int(chunk_seconds * fs)
        max_chunks = int(max(self.settings.voice_max_record_seconds, 1.0) / chunk_seconds)
        min_speech_chunks = int(max(self.settings.voice_min_speech_seconds, 0.2) / chunk_seconds)
        silence_timeout_chunks = int(max(self.settings.voice_silence_timeout_seconds, 0.3) / chunk_seconds)
        end_speech_min_chunks = max(2, int(min_speech_chunks * 0.75))

        threshold = max(self.noise_floor * self.settings.voice_noise_gate_multiplier, 0.006)

        chunks = []
        speech_chunks = 0
        silence_after_speech = 0
        started = False

        with self.sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype="float32",
            device=self.settings.voice_input_device,
            blocksize=chunk_frames,
        ) as stream:
            for _ in range(max_chunks):
                if self.interrupt_event.is_set():
                    return None
                chunk, _ = stream.read(chunk_frames)
                mono = chunk[:, 0]
                energy = self._rms(mono)

                if energy > threshold:
                    started = True
                    speech_chunks += 1
                    silence_after_speech = 0
                    chunks.append(chunk)
                else:
                    if started:
                        chunks.append(chunk)
                        silence_after_speech += 1
                        # Corta antes cuando ya hay voz suficiente y silencio continuo.
                        enough_speech_collected = speech_chunks >= end_speech_min_chunks
                        if silence_after_speech >= silence_timeout_chunks and enough_speech_collected:
                            break

        if speech_chunks < min_speech_chunks:
            return None

        audio = self.np.concatenate(chunks, axis=0)
        audio = self._apply_basic_noise_suppression(audio)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        self.sf.write(tmp.name, audio, fs)
        return tmp.name

    def _transcribe_once(
        self,
        file_path: str,
        beam_size: int,
        best_of: int,
        temperature: float,
        no_speech_threshold: float,
        log_prob_threshold: float,
        vad_min_silence_ms: int,
    ) -> str:
        segments, _ = self.stt_model.transcribe(
            file_path,
            language=self.settings.voice_language,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            vad_filter=self.settings.voice_stt_use_vad_filter,
            vad_parameters={"min_silence_duration_ms": vad_min_silence_ms},
            condition_on_previous_text=False,
            initial_prompt=(
                "Conversacion en espanol. Nombres frecuentes: Daniel y Zentris. "
                "Transcribe con precision pronunciacion lenta."
            ),
        )
        return " ".join(segment.text for segment in segments).strip()

    def transcribe_file(self, file_path: str) -> str:
        primary = self._transcribe_once(
            file_path=file_path,
            beam_size=self.settings.voice_stt_beam_size,
            best_of=self.settings.voice_stt_best_of,
            temperature=self.settings.voice_stt_temperature,
            no_speech_threshold=self.settings.voice_stt_no_speech_threshold,
            log_prob_threshold=self.settings.voice_stt_logprob_threshold,
            vad_min_silence_ms=self.settings.voice_vad_min_silence_ms,
        )

        # Rescate para frases lentas o recortadas.
        if len(primary.split()) <= 2:
            rescue = self._transcribe_once(
                file_path=file_path,
                beam_size=max(self.settings.voice_stt_beam_size + 2, 8),
                best_of=max(self.settings.voice_stt_best_of + 2, 6),
                temperature=0.2,
                no_speech_threshold=min(self.settings.voice_stt_no_speech_threshold, 0.30),
                log_prob_threshold=min(self.settings.voice_stt_logprob_threshold, -1.8),
                vad_min_silence_ms=max(int(self.settings.voice_vad_min_silence_ms * 0.8), 250),
            )
            if len(rescue) > len(primary):
                self.logger.info("STT rescate aplicado")
                return rescue

        return primary

    async def generate_reply(self, user_id: str, user_message: str) -> str:
        history = self.core.user_memory.get(user_id, "")
        preferences = self.core.update_response_preferences(user_id, user_message)
        quick = self.core.fast_response(user_message)
        if quick:
            return self.core.finalize_reply(quick, preferences, user_message=user_message, history=history)

        tool_reply = await self.core.try_tool_response(user_message)
        if tool_reply:
            tool_reply = self.core.finalize_reply(tool_reply, preferences, user_message=user_message, history=history)
            self.core.persist_text_interaction(user_id, user_message, tool_reply)
            return tool_reply

        profile = self.core.get_profile_for_prompt(user_id)
        style = self.core.get_user_style(user_id)
        relevant_memory = ""
        if len(user_message.split()) > 2:
            relevant_memory = self.core.vector_memory.get_relevant_memories(user_id, user_message)

        prompt = self.core.build_text_prompt(user_message, profile, relevant_memory, history, style, preferences)
        model = self.settings.fast_model if self.settings.voice_force_fast_model else self.core.choose_model(user_message)
        self.logger.info("Generando respuesta con modelo: %s", model)

        try:
            reply = await self.ollama.generate_text(prompt, model)
        except Exception:
            if model != self.settings.balanced_model:
                reply = await self.ollama.generate_text(prompt, self.settings.balanced_model)
            else:
                raise

        reply = self.core.sanitize_reply(profile, reply.strip(), user_message=user_message)
        reply = await enforce_spanish(
            reply,
            self.settings.balanced_model,
            self.ollama.generate_text,
            self.logger,
        )
        reply = self.core.finalize_reply(reply, preferences, user_message=user_message, history=history)

        should_repair = (
            reply
            and not self.core.is_reply_on_topic(user_message, reply)
            and len(user_message.split()) > 4
            and not preferences.get("use_chilean", False)
        )
        if should_repair:
            repair_prompt = self.core.build_precision_prompt(
                user_message,
                profile,
                relevant_memory,
                history,
                style,
                preferences,
            )
            reply = await self.ollama.generate_text(repair_prompt, self.settings.balanced_model)
            reply = self.core.sanitize_reply(profile, reply.strip(), user_message=user_message)
            reply = await enforce_spanish(
                reply,
                self.settings.balanced_model,
                self.ollama.generate_text,
                self.logger,
            )
            reply = self.core.finalize_reply(reply, preferences, user_message=user_message, history=history)

        if not reply:
            reply = "No pude responder."

        self.core.persist_text_interaction(user_id, user_message, reply)
        if len(user_message.split()) > 2:
            self.core.vector_memory.save_memory(user_id, user_message)

        await self.core.extract_user_info_async(user_id, user_message, run_llm=self.settings.voice_profile_extract_llm)
        return reply

    def run(self) -> None:
        self.logger.info("Modo local de voz activo")
        self.logger.info("Habla despues del mensaje 'Escuchando...' y espera la respuesta")
        self.logger.info("Presiona cualquier tecla para interrumpir el turno actual")
        self._start_keyboard_listener()

        user_id = self.settings.allowed_chat_id

        try:
            while True:
                try:
                    self.interrupt_event.clear()
                    print("Escuchando...")
                    audio_path = self.record_audio_file()
                    if not audio_path:
                        continue
                    try:
                        text = self.transcribe_file(audio_path)
                    finally:
                        try:
                            os.remove(audio_path)
                        except Exception:
                            pass

                    if not text:
                        continue

                    if self.interrupt_event.is_set():
                        continue

                    text = self.normalize_transcription(text)

                    print(f"Tu: {text}")
                    if text.lower() in {"salir", "exit", "stop"}:
                        self.speak("Cerrando modo local")
                        break

                    print("Pensando...")
                    try:
                        reply = asyncio.run(
                            asyncio.wait_for(
                                self.generate_reply(user_id, text),
                                timeout=max(self.settings.request_timeout, 20),
                            )
                        )
                    except asyncio.TimeoutError:
                        reply = "Tarde demasiado en responder. Intenta de nuevo con una pregunta mas corta."

                    if self.interrupt_event.is_set():
                        continue

                    self.speak(reply)

                except KeyboardInterrupt:
                    self.speak("Interrumpido")
                    break
                except Exception as exc:
                    self.logger.exception("Error en modo local de voz")
                    print(f"Error: {exc}")
        finally:
            self._stop_keyboard_listener()
