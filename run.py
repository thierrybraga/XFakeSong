import os
# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import uvicorn
from gradio_app import create_unified_app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4443))
    app = create_unified_app(port)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300, # Aumentado para evitar desconexões em SSE/Heartbeat
        ws_ping_interval=None,  # Desabilitar ping automático se estiver causando resets
        ws_ping_timeout=None
    )
