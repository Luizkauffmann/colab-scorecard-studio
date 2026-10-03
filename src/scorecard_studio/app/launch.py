"""
Start the binning app from a notebook cell.

In Colab the page is served through the kernel-port proxy
(``google.colab.output.serve_kernel_port_as_iframe``): it stays inside your
Colab session, no public tunnel, nothing leaves the runtime. Outside Colab it
is shown in an IFrame from ``http://127.0.0.1:<port>``.
"""

from __future__ import annotations

import glob
import json
import os
import socket
import threading
from typing import Any, Dict, List, Optional

import pandas as pd

from ..screening import target_sample
from .server import create_app
from .session import CONFIG_FILE, AppError, BinningSession


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


class BinningApp:
    """A running app. One :class:`BinningSession` per target: ``session`` is the
    primary target's, ``sessions[target]`` any other; ``engine`` is the primary
    target's fitted :class:`~scorecard_studio.BinningEngine`."""

    def __init__(self, sessions, port: Optional[int] = None, host: str = "127.0.0.1",
                 primary: Optional[str] = None):
        from werkzeug.serving import make_server
        if isinstance(sessions, BinningSession):
            sessions = {sessions.target: sessions}
        self.sessions: Dict[str, BinningSession] = dict(sessions)
        self.primary = primary or next(iter(self.sessions))
        self.port = port or _free_port()
        self.host = host
        self._server = make_server(host, self.port, create_app(self.sessions, self.primary), threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name=f"binning-app-{self.port}")
        self._thread.start()

    # ------------------------------------------------------------------

    @property
    def session(self) -> BinningSession:
        return self.sessions[self.primary]

    @property
    def targets(self) -> List[str]:
        return list(self.sessions)

    @property
    def engine(self):
        return self.session.engine

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def show(self, height: int = 860):
        """Display the app in the cell output."""
        if _in_colab():
            from google.colab import output  # type: ignore
            output.serve_kernel_port_as_iframe(self.port, height=height)
        else:
            from IPython.display import IFrame, display
            display(IFrame(self.url, width="100%", height=height))

    def open_in_tab(self):
        """Open the app in a new browser tab (more room for wide tables)."""
        if _in_colab():
            from google.colab import output  # type: ignore
            output.serve_kernel_port_as_window(self.port, anchor_text="Open the binning app in a new tab")
        else:
            print(self.url)

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=5)

    # Pass-throughs so the notebook doesn't need to know about the session.
    def build_output(self, variables=None, target: Optional[str] = None) -> pd.DataFrame:
        return self.sessions[target or self.primary].build_output(variables)

    def save_output(self, variables=None, target: Optional[str] = None) -> Dict[str, Any]:
        return self.sessions[target or self.primary].save_output(variables)

    @property
    def included(self) -> List[str]:
        return self.session.included

    def __repr__(self) -> str:
        return f"BinningApp(port={self.port}, variables={len(self.session.vars)}, included={len(self.included)})"


def find_saved_configs(output_dir: str) -> List[str]:
    """``binning_config.json`` files under ``output_dir``, newest first."""
    paths = glob.glob(os.path.join(output_dir, "run_*", "**", CONFIG_FILE), recursive=True)
    return sorted(paths, key=os.path.getmtime, reverse=True)


def launch_app(train: pd.DataFrame, target: str, *, store=None, resume: bool = True,
               show: bool = True, port: Optional[int] = None, height: int = 860,
               alt_targets: Optional[List[str]] = None, screens: Optional[Dict[str, Any]] = None,
               full: Optional[pd.DataFrame] = None, sample: Optional[pd.Series] = None,
               **session_kwargs) -> BinningApp:
    """Create one :class:`BinningSession` per target, restore earlier work,
    start the server and display the app.

    ``alt_targets``: other candidate targets declared in the config; the app
    gets a target selector. For each target only rows with a known outcome
    are used, and the other target columns are removed (never predictors).
    ``screens``: ``{target: ScreeningResult}`` (or pass ``screen=`` for the
    primary target only).

    ``store`` (a :class:`~scorecard_studio.store.RunStore`) sets where every
    change is saved (``04_binning/<target>/``). With ``resume=True`` the newest
    saved binning for *the same Train sample* in any earlier run is reloaded,
    so a runtime reset doesn't lose work. Configs saved for other samples are
    skipped (bins edited on different rows aren't the same bins).
    """
    targets = [target, *(alt_targets or [])]
    screens = dict(screens or {})
    if "screen" in session_kwargs:
        screens.setdefault(target, session_kwargs.pop("screen"))
    if (full is None) != (sample is None):
        raise ValueError("Pass both `full` and `sample`, or neither.")
    saved = find_saved_configs(store.output_dir) if (resume and store is not None) else []

    sessions: Dict[str, BinningSession] = {}
    for t in targets:
        kw = dict(session_kwargs)
        if store is not None:
            kw.setdefault("save_dir", os.path.join(store.folder("binning"), _safe(t)))
        f = smp = None
        if full is not None:
            keep = full[t].notna()
            f = target_sample(full, t, targets)
            smp = sample.loc[keep]
        sess = BinningSession(target_sample(train, t, targets), t, full=f, sample=smp,
                              screen=screens.get(t), **kw)
        for path in saved:
            try:
                with open(path) as fh:
                    skipped = sess.load(json.load(fh))
            except (AppError, ValueError, KeyError, json.JSONDecodeError):
                continue
            print(f"[{t}] resumed binning saved at {path}" + (f" (not restored: {skipped})" if skipped else ""))
            break
        sess.save()
        sessions[t] = sess

    app = BinningApp(sessions, port=port, primary=target)
    if show:
        app.show(height=height)
    return app


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name)) or "target"
