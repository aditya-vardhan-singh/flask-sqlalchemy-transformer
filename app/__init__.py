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

    # Import and register blueprints here (we'll add routes later)
    from app.routes import routes
    app.register_blueprint(routes.bp) # Assuming routes will be in a Blueprint

    return app

from app.models import models # Import models so SQLAlchemy can find them
