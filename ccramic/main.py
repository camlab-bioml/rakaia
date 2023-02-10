import os
import subprocess


def run_app():
    subprocess.run(["streamlit", "run", os.path.join(os.path.dirname(__file__), "app.py")])


if __name__ == "__main__":
    run_app()
