import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
                              'postgresql://user:password@localhost:5432/mydatabase'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DEBUG = True

class DevelopmentConfig(Config):
    DEBUG = True