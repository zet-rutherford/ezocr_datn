# -*- coding: utf-8 -*-
"""The app module, containing the app factory function."""

from flask import Flask, Blueprint, jsonify

# from app import commands
from .extensions import config, logger
# from app.controller.invoice_reader.auth_jwt import JWTError

def create_app():
    """An application factory, as explained here:
    http://flask.pocoo.org/docs/patterns/appfactories/.
    """
    initialize()
    app = Flask(__name__)
    app.url_map.strict_slashes = False
    app.config.from_object(config)
    logger.debug('Application instance created')
    # register_extensions(app)
    register_blueprints(app)
    register_error_handlers(app)
    register_shell_context(app)
    # register_commands(app)
    return app
 

health = Blueprint('health', __name__)

@health.route('/ping', methods=['GET'])
def ping():
    return jsonify(success=True, data="service is healthy")

def register_blueprints(app):
    """Register Flask blueprints."""

    # API routes
    from app.controller.invoice_reader import routes
    app.register_blueprint(routes.blueprint, url_prefix='/extraction')
    app.register_blueprint(health)

def register_error_handlers(app):
    def handle_jwt_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    # app.errorhandler(JWTError)(handle_jwt_error)

def register_shell_context(app):
    """Register shell context objects."""
    pass


# def register_commands(app):
#     """Register Click commands."""
#     app.cli.add_command(commands.test)
#     app.cli.add_command(commands.lint)
#     app.cli.add_command(commands.clean)
#     app.cli.add_command(commands.urls)


def initialize():
    from app.controller.invoice_reader.routes import init
    init()