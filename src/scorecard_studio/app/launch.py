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
    """A running app: ``session`` holds the bins, ``engine`` is the fitted
    :class:`~scorecard_studio.BinningEngine` for the next stages."""

    def __init__(self, session: BinningSession, port: Optional[int] = None, host: str = "127.0.0.1"):
        from werkzeug.serving import make_server
        self.session = session
        self.port = port or _free_port()
        self.host = host
        self._server = make_server(host, self.port, create_app(session), threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name=f"binning-app-{self.port}")
        self._thread.start()

    # ------------------------------------------------------------------

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
    def build_output(self, variables=None) -> pd.DataFrame:
        return self.session.build_output(variables)

    def save_output(self, variables=None) -> Dict[str, Any]:
        return self.session.save_output(variables)

    @property
    def included(self) -> List[str]:
        return self.session.included

    def __repr__(self) -> str:
        return f"BinningApp(port={self.port}, variables={len(self.session.vars)}, included={len(self.included)})"


def find_saved_configs(output_dir: str) -> List[str]:
    """``binning_config.json`` files under ``output_dir``, newest first."""
    paths = glob.glob(os.path.join(output_dir, "run_*", "*", CONFIG_FILE))
    return sorted(paths, key=os.path.getmtime, reverse=True)


def launch_app(train: pd.DataFrame, target: str, *, store=None, resume: bool = True,
               show: bool = True, port: Optional[int] = None, height: int = 860,
               **session_kwargs) -> BinningApp:
    """Create a :class:`BinningSession`, restore earlier work, start the server
    and display the app.

    ``store`` (a :class:`~scorecard_studio.store.RunStore`) sets where every
    change is saved (``04_binning/``). With ``resume=True`` the newest saved
    binning for *this same Train sample* in any earlier run is reloaded, so a
    runtime reset doesn't lose work. Configs saved for other samples are
    skipped (bins edited on different rows aren't the same bins).
    """
    if store is not None:
        session_kwargs.setdefault("save_dir", store.folder("binning"))
    session = BinningSession(train, target, **session_kwargs)

    if resume and store is not None:
        for path in find_saved_configs(store.output_dir):
            try:
                with open(path) as fh:
                    skipped = session.load(json.load(fh))
            except (AppError, ValueError, KeyError, json.JSONDecodeError):
                continue
            print(f"Resumed binning saved at {path}" + (f" (not restored: {skipped})" if skipped else ""))
            break
    session.save()

    app = BinningApp(session, port=port)
    if show:
        app.show(height=height)
    return app
