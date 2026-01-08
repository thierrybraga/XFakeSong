from flask import Blueprint, jsonify

api = Blueprint('api', __name__, url_prefix='/api/v1')


@api.route('/system/bootstrap')
def bootstrap():
    return jsonify({
        "status": "ok",
        "message": "System operational",
        "has_active_user": True,
        "first_sync_completed": True,
        "sync_in_progress": False
    })
