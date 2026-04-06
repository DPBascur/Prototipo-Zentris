from app.bootstrap import create_app


def main() -> None:
    app = create_app()
    print("Zentris optimizado corriendo...")
    app.run_polling()


if __name__ == "__main__":
    main()