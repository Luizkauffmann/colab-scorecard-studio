"""
Flask routes for the binning app.

The frontend only uses *relative* URLs (``api/...``), so the same page works
behind Colab's kernel-port proxy, in a full browser tab, or on localhost.
Every response is JSON; errors come back as ``{"error": message}`` with
status 400 (user action rejected) or 500 (bug), and the UI shows them.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory

from .session import AppError, BinningSession, _json_safe

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app(sessions, primary: Optional[str] = None) -> Flask:
    """``sessions``: one :class:`BinningSession`, or ``{target: session}``.
    Every route takes an optional ``target`` (query string or JSON body)."""
    if isinstance(sessions, BinningSession):
        sessions = {sessions.target: sessions}
    primary = primary or next(iter(sessions))
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False

    def body():
        return request.get_json(silent=True) or {}

    def sess() -> BinningSession:
        t = request.args.get("target") or (body().get("target") if request.is_json else None) or primary
        if t not in sessions:
            raise AppError(f"Unknown target '{t}'.")
        return sessions[t]

    def ok(payload):
        if isinstance(payload, dict) and isinstance(payload.get("state"), dict):
            payload["state"].update(targets=list(sessions), primary=primary)
        return jsonify(_json_safe(payload))

    @app.errorhandler(AppError)
    def _app_error(exc):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(Exception)
    def _unexpected(exc):  # pragma: no cover - surfaced in the UI and the cell log
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
        s = sess()
        with s.lock:
            return ok({"state": s.state()})

    @app.get("/api/variable")
    def variable():
        s = sess()
        name = request.args.get("name", "")
        return ok({"variable": s.variable_payload(name), "state": s.state()})

    @app.post("/api/fit")
    def fit():
        b = body()
        params = {k: b[k] for k in ("max_bins", "monotonic", "min_bin_size",
                                    "min_bin_n_event", "cat_cutoff") if k in b}
        return ok(sess().fit(b.get("name", ""), **params))

    @app.post("/api/reset")
    def reset():
        return ok(sess().reset(body().get("name", "")))

    @app.post("/api/cutoffs")
    def cutoffs():
        b = body()
        return ok(sess().set_cutoffs(b.get("name", ""), b.get("cutoffs", [])))

    @app.post("/api/split")
    def split():
        b = body()
        return ok(sess().split(b.get("name", ""), int(b.get("group", 0)), b.get("at")))

    @app.post("/api/merge")
    def merge():
        b = body()
        return ok(sess().merge(b.get("name", ""), b.get("groups", [])))

    @app.post("/api/fixed")
    def fixed():
        b = body()
        to_int = lambda v: None if v in (None, "", "separate") else int(v)
        return ok(sess().set_fixed(b.get("name", ""), missing_to=to_int(b.get("missing_to")),
                                   special_to=to_int(b.get("special_to"))))

    @app.post("/api/categories")
    def categories():
        b = body()
        return ok(sess().set_categories(b.get("name", ""), b.get("assignments", {})))

    @app.post("/api/include")
    def include():
        b = body()
        return ok(sess().set_included(b.get("name", ""), bool(b.get("included"))))

    @app.post("/api/output")
    def output():
        s = sess()
        return ok({"output": s.save_output(), "state": s.state()})

    return app
