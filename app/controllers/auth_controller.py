from flask import Blueprint

auth = Blueprint('auth', __name__, url_prefix='/')


@auth.route('/init-root')
def init_root():
    return "Init Root Page"
