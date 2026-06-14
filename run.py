import subprocess
import sys
import config

def check_python_version():
    """
    Vérifie que la version de Python est compatible avec ChromaDB (3.13 ou moins)
    """
    if sys.version_info.major == 3 and sys.version_info.minor >= 14:
        print("⚠️  AVERTISSEMENT : Vous utilisez Python " + ".".join(map(str, sys.version_info[:3])))
        print("ChromaDB et Pydantic V1 sont actuellement incompatibles avec Python 3.14+.")
        print("Veuillez utiliser Python 3.13 ou une version inférieure.")
        print("-" * 50)

def run_app():
    """
    Lance l'interface Streamlit en utilisant les paramètres définis dans config.py
    """
    cmd = [
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        config.SERVER_ADDRESS,
        "--server.port",
        str(config.SERVER_PORT)
    ]

    print(f"🚀 Lancement de RPG Oracle sur http://{config.SERVER_ADDRESS}:{config.SERVER_PORT}")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Arrêt du serveur.")
    except Exception as e:
        print(f"❌ Erreur lors du lancement : {e}")
        sys.exit(1)

if __name__ == "__main__":
    config.check_config()
    check_python_version()
    run_app()
