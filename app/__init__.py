from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import Config
import os

db = SQLAlchemy()
migrate = Migrate()

def create_app(config_class="config.DevelopmentConfig"):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Register blueprints
    from app.routes.routes import bp as main_bp
    app.register_blueprint(main_bp)

    from app.routes.chatbot import bp as chatbot_bp
    app.register_blueprint(chatbot_bp)

    return app

from app.models import models # Import models so SQLAlchemy can find them
