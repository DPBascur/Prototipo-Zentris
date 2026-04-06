import threading

from app.bootstrap import create_runtime


def main() -> None:
    settings, app, voice_runner = create_runtime()

    if settings.app_mode == "telegram":
        if app is None:
            raise RuntimeError("No se pudo inicializar la app de Telegram")
        print("Zentris optimizado corriendo en modo telegram...")
        app.run_polling()
        return

    if settings.app_mode == "local":
        if voice_runner is None:
            raise RuntimeError("No se pudo inicializar el runner local de voz")
        print("Zentris optimizado corriendo en modo local...")
        voice_runner.run()
        return

    if app is None or voice_runner is None:
        raise RuntimeError("No se pudo inicializar el modo hybrid")

    print("Zentris optimizado corriendo en modo hybrid...")
    voice_thread = threading.Thread(target=voice_runner.run, daemon=True)
    voice_thread.start()
    app.run_polling()


if __name__ == "__main__":
    main()