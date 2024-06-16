import os
from dotenv import load_dotenv


def load_environment_variables(env: str):
    load_dotenv(dotenv_path=f"env/.env.{env}", verbose=True) or load_dotenv()


environment = os.getenv("ENV", "local")
load_environment_variables(environment)
