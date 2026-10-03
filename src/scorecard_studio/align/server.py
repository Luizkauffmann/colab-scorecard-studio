"""Flask routes for the alignment app (JSON only, relative URLs)."""

from __future__ import annotations

import os
import traceback

from flask import Flask, jsonify, request, send_from_directory

from ..app.session import _json_safe
from .session import AlignError, AlignSession

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_align_app(session: AlignSession) -> Flask:
    app = Flask(__name__, static_folder=None)

    def body():
        return request.get_json(silent=True) or {}

    def ok(payload):
        return jsonify(_json_safe(payload))

    def everything():
        return {"state": session.state(), "strategy": session.strategy()}

    @app.errorhandler(AlignError)
    def _err(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(Exception)
    def _unexpected(exc):  # pragma: no cover
        traceback.print_exc()
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC_DIR, name)

    @app.get("/api/state")
    def state():
        return ok(everything())

    @app.post("/api/data")
    def data():
        b = body()
        session.set_data(b.get("dataset") or None, b.get("target") or None)
        return ok(everything())

    @app.post("/api/settings")
    def settings():
        session.set_settings(**body())
        return ok(everything())

    @app.get("/api/scorecard")
    def scorecard():
        return ok(session.scorecard_rows())

    @app.post("/api/points")
    def points():
        b = body()
        session.set_points(b.get("variable", ""), b.get("group"), b.get("points"), b.get("reason", ""))
        return ok({**everything(), "scorecard": session.scorecard_rows()})

    @app.post("/api/reset")
    def reset():
        b = body()
        session.reset_points(b.get("variable"), b.get("group"))
        return ok({**everything(), "scorecard": session.scorecard_rows()})

    @app.get("/api/stats")
    def stats():
        return ok(session.stats(request.args.get("sample") or None))

    @app.post("/api/finalize")
    def finalize():
        info = session.finalize()
        preview = session.final_table().head(15)
        return ok({"final": info, "preview": {"columns": list(preview.columns),
                                               "rows": preview.astype(object).where(preview.notna(), None).values.tolist()},
                   "state": session.state()})

    @app.get("/api/code")
    def code():
        a = request.args
        return ok({"code": session.code(a.get("lang", "python"), a.get("dialect", "standard"),
                                        a.get("table", "input_table"))})

    @app.post("/api/code/save")
    def save_code():
        b = body()
        return ok({"path": session.save_code(b.get("lang", "python"), b.get("dialect", "standard"),
                                             b.get("table", "input_table"))})

    return app
