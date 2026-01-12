from flask import Flask
# from app.core.config.settings import SystemConfig
from app.extensions import db, migrate, login_manager
from app.controllers.api_controller import api
from app.controllers.auth_controller import auth
from app.controllers.main_controller import main
from app.domain.models.user import User


def create_app(config=None):
    app = Flask(__name__)

    # Load config (basic setup for now)
    # system_config = SystemConfig()
    app.config['SECRET_KEY'] = 'dev-key'
    # Use config from SystemConfig if available, or defaults
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    # Register user loader
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Register blueprints
    app.register_blueprint(api)
    app.register_blueprint(auth)
    app.register_blueprint(main)

    return app
