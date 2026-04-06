import sounddevice as sd


def main() -> None:
    devices = sd.query_devices()
    default_input, default_output = sd.default.device

    print("Dispositivos de audio disponibles")
    print("=" * 40)
    print(f"Entrada por defecto: {default_input}")
    print(f"Salida por defecto: {default_output}")
    print()

    found_input = False
    for index, dev in enumerate(devices):
        max_input = int(dev.get("max_input_channels", 0))
        if max_input <= 0:
            continue

        found_input = True
        marker = " (default)" if index == default_input else ""
        name = dev.get("name", "Sin nombre")
        hostapi = dev.get("hostapi", "?")
        rate = dev.get("default_samplerate", "?")

        print(f"[{index}] {name}{marker}")
        print(f"    hostapi={hostapi} input_channels={max_input} sample_rate={rate}")

    if not found_input:
        print("No se detectaron dispositivos de entrada (micrófono).")


if __name__ == "__main__":
    main()
