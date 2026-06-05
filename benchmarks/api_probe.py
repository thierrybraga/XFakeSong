"""Teste de sistema da API REST via FastAPI TestClient (sem rede).

Valida que o app sobe e que os endpoints respondem, medindo a latência por
endpoint. Lê a superfície real (OpenAPI) e sonda os GET sem parâmetros — assim
não depende de caminhos chumbados que possam mudar.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger("benchmark")

# Endpoints informativos têm prioridade na amostragem.
_PRIORITY_HINTS = ("health", "version", "info", "status", "metrics",
                   "architectures", "models")


def run_api_probe(max_endpoints: int = 8) -> Dict[str, Any]:
    """Sobe o app via TestClient e sonda endpoints GET sem parâmetros."""
    try:
        from fastapi.testclient import TestClient

        from app.main_fastapi import app
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "error": str(e)}

    endpoints: List[Dict[str, Any]] = []
    n_routes = len([r for r in app.routes if getattr(r, "methods", None)])

    try:
        with TestClient(app) as client:
            spec = client.get("/openapi.json")
            paths: Dict[str, Any] = {}
            if spec.status_code == 200:
                try:
                    paths = spec.json().get("paths", {})
                except Exception:
                    paths = {}

            # GET sem parâmetros de caminho
            candidates = [
                p for p, ops in paths.items()
                if "{" not in p and "get" in {m.lower() for m in ops}
            ]
            candidates.sort(
                key=lambda p: 0 if any(h in p for h in _PRIORITY_HINTS) else 1
            )
            probe = ["/openapi.json"] + candidates[:max_endpoints]

            for path in probe:
                t0 = time.perf_counter()
                try:
                    resp = client.get(path)
                    endpoints.append({
                        "method": "GET",
                        "path": path,
                        "status_code": int(resp.status_code),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                        "ok": 200 <= resp.status_code < 300,
                    })
                except Exception as e:  # noqa: BLE001
                    endpoints.append({
                        "method": "GET", "path": path, "error": str(e)[:160],
                        "ok": False,
                    })
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e), "n_routes": n_routes}

    n_ok = sum(1 for e in endpoints if e.get("ok"))
    lat = [e["latency_ms"] for e in endpoints if "latency_ms" in e]
    return {
        "status": "ok" if n_ok else "degraded",
        "n_routes": n_routes,
        "n_probed": len(endpoints),
        "n_2xx": n_ok,
        "latency_ms_median": round(float(sorted(lat)[len(lat) // 2]), 2) if lat
        else None,
        "endpoints": endpoints,
    }


__all__ = ["run_api_probe"]
