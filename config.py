import os

class Config:
    SECRET_KEY = os.getenv("DARTS_SQL_KEY", "default_secret_key")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "postgresql://localhost/darts_db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False