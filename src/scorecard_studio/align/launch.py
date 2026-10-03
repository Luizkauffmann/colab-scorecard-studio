"""Start the alignment app from a notebook cell (same serving as the binning app)."""

from __future__ import annotations

import glob
import json
import os
import threading
from typing import Optional

import pandas as pd

from ..app.launch import _free_port, _in_colab
from ..scorecard import Scorecard
from .server import create_align_app
from .session import STATE_FILE, AlignError, AlignSession


class AlignApp:
    def __init__(self, session: AlignSession, port: Optional[int] = None, host: str = "127.0.0.1"):
        from werkzeug.serving import make_server
        self.session = session
        self.port = port or _free_port()
        self.host = host
        self._server = make_server(host, self.port, create_align_app(session), threaded=True)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name=f"align-app-{self.port}")
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def scorecard(self) -> Scorecard:
        """The scorecard as edited in the app (points, scale, cutoff)."""
        return self.session.card

    def show(self, height: int = 900):
        if _in_colab():
            from google.colab import output  # type: ignore
            output.serve_kernel_port_as_iframe(self.port, height=height)
        else:
            from IPython.display import IFrame, display
            display(IFrame(self.url, width="100%", height=height))

    def open_in_tab(self):
        if _in_colab():
            from google.colab import output  # type: ignore
            output.serve_kernel_port_as_window(self.port, anchor_text="Open the alignment app in a new tab")
        else:
            print(self.url)

    def stop(self):
        self._server.shutdown()
        self._thread.join(timeout=5)

    def finalize(self):
        return self.session.finalize()

    def final_table(self) -> pd.DataFrame:
        return self.session.final_table()

    def code(self, lang: str = "python", dialect: str = "standard", table: str = "input_table") -> str:
        return self.session.code(lang, dialect, table)


def launch_alignment(card: Scorecard, data: pd.DataFrame, target: str, *, store=None, resume: bool = True,
                     show: bool = True, port: Optional[int] = None, height: int = 900,
                     **session_kwargs) -> AlignApp:
    """Open the alignment app on ``card`` and ``data`` (rows with the raw
    variables, the target and, via ``sample=``, their sample). With ``store``,
    state is saved to ``08_alignment/`` and the newest state saved for the same
    model in any earlier run is reloaded."""
    if store is not None:
        session_kwargs.setdefault("save_dir", store.folder("alignment"))
    session = AlignSession(card, data, target, **session_kwargs)
    if resume and store is not None:
        paths = sorted(glob.glob(os.path.join(store.output_dir, "run_*", "**", STATE_FILE), recursive=True),
                       key=os.path.getmtime, reverse=True)
        for p in paths:
            try:
                with open(p) as fh:
                    session.load(json.load(fh))
            except (AlignError, ValueError, KeyError, json.JSONDecodeError):
                continue
            print(f"Resumed alignment saved at {p}")
            break
    session.save()
    app = AlignApp(session, port=port)
    if show:
        app.show(height=height)
    return app
