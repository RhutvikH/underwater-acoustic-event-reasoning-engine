"""Presentation demo: live swarm twin over HTTP JSON (GUI + Unity)."""

from __future__ import annotations

import json
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from urllib.parse import parse_qs, urlparse

from uaere.data.wavutil import wav_bytes
from uaere.models.fish_head import DEFAULT_FISH_CHECKPOINT
from uaere.swarm.field import SwarmField, tick_to_json

STATIC = Path(__file__).with_name("static")


class DemoState:
    def __init__(self, field: SwarmField, period_s: float = 0.85) -> None:
        self.field = field
        self.period_s = period_s
        self.lock = threading.Lock()
        self.tick = None
        self.paused = False
        self._stop = False

    def loop(self) -> None:
        self.field.warmup()
        while not self._stop:
            if not self.paused:
                t = self.field.step()
                with self.lock:
                    self.tick = t
            time.sleep(self.period_s)

    def json(self) -> bytes:
        with self.lock:
            if self.tick is None:
                payload = {
                    "t": 0,
                    "nodes": [],
                    "links": [],
                    "kpis": {},
                    "source": {},
                    "environment": {},
                    "scenario": self.field.scenario,
                    "model": self.field.model,
                }
            else:
                payload = tick_to_json(self.tick)
        payload["paused"] = self.paused
        return json.dumps(payload, default=float).encode("utf-8")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, state: DemoState, *args, **kwargs) -> None:
        self.state = state
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        pass

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/audio"):
            node = (parse_qs(urlparse(self.path).query).get("node") or ["n0"])[0]
            audio = self.state.field._audio.get(node)
            if audio is None:
                self.send_error(404, "no window yet")
                return
            fs = int(getattr(self.state.field, "playback_fs", None) or 16_000)
            body = wav_bytes(audio, fs)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/api/state"):
            body = self.state.json()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/pause":
            self.state.paused = not self.state.paused
            self._ok({"paused": self.state.paused})
            return
        if self.path == "/api/step":
            t = self.state.field.step()
            with self.state.lock:
                self.state.tick = t
            self._ok({"ok": True})
            return
        self.send_error(404)

    def _ok(self, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DemoServer:
    def __init__(self, field: SwarmField, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.state = DemoState(field)
        self.host = host
        self.port = port
        self.httpd = ThreadingHTTPServer((host, port), partial(Handler, self.state))

    def start(self) -> None:
        threading.Thread(target=self.state.loop, daemon=True).start()
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.state._stop = True
        self.httpd.shutdown()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"


def resolve_model(model: str, checkpoint: str | None) -> str:
    """auto prefers the species checkpoint, then the fish checkpoint, then the ship logistic.

    Oracle stays opt-in. A checkpoint path is classified by its ``kind`` field.
    """
    if model != "auto":
        return model
    from pathlib import Path

    import joblib

    from uaere.models.species_head import DEFAULT_SPECIES_CHECKPOINT, KINDS

    path = Path(checkpoint) if checkpoint else None
    if path is None:
        if DEFAULT_SPECIES_CHECKPOINT.exists():
            path = DEFAULT_SPECIES_CHECKPOINT
        elif Path(DEFAULT_FISH_CHECKPOINT).exists():
            path = Path(DEFAULT_FISH_CHECKPOINT)
        else:
            print(
                "No species or fish checkpoint under artifacts/models. "
                "Demo will use the ship logistic head. "
                "Train with `uaere train --dataset species --root data/species`, "
                "or pass --model oracle for the twin illustration (not a paper result)."
            )
            return "logistic"
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Demo will not train a ship model in its place."
        )
    kind = joblib.load(path).get("kind")
    if kind in KINDS:
        return "species"
    if kind == "fish_logistic_mel":
        return "fish"
    return "logistic"


def serve_demo(
    n_nodes: int = 8,
    scenario: str = "pam_field",
    seed: int = 0,
    port: int = 8765,
    model: str = "auto",
    replay_root: str | None = None,
    checkpoint: str | None = None,
) -> DemoServer:
    model = resolve_model(model, checkpoint)
    field = SwarmField(
        n_nodes=n_nodes,
        scenario=scenario,
        seed=seed,
        model=model,
        replay_root=replay_root,
        checkpoint=checkpoint,
    )
    srv = DemoServer(field, port=port)
    srv.start()
    return srv
