from flask import Flask
from flask_restful import Api
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)
    # api = Api(app)
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'postgresql://postgres:2004@localhost:5432/plag_search_db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Register blueprints
    from app.blueprints.plag_check import plag_check_bp
    app.register_blueprint(plag_check_bp)
    
    return app