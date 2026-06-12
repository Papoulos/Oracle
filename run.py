import subprocess
import sys
import config

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
    run_app()
