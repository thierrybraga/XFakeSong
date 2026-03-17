from flask import Flask, jsonify, redirect


def create_app(config=None):
    # Flask app mantido apenas para compatibilidade se houver rotas legadas
    # mas o banco agora é FastAPI nativo.
    app = Flask(__name__)

    @app.get("/")
    def home():
        return redirect("/gradio/", code=302)

    @app.get("/api/v1/system/bootstrap")
    def bootstrap():
        return jsonify({"status": "ok"})

    return app
