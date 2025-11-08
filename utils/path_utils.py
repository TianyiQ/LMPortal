import os
import sys

import dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
for lev in range(5):
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    libpath = os.path.join(current_dir, "lib")
    if os.path.exists(libpath):
        if libpath not in sys.path:
            sys.path.append(libpath)

        # Load API keys from .env
        env_path = os.path.join(libpath, "safety_tooling", ".env")
        if os.path.exists(env_path):
            dotenv.load_dotenv(env_path)
            print(f"Loaded API keys from {env_path}")

    current_dir = os.path.dirname(current_dir)
