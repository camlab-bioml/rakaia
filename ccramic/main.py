import os
import subprocess
from app.app import app
import dash


def main():
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
