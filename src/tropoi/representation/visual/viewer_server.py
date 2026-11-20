"""Read-only loopback transport for the local Viewer 2 application."""
from __future__ import annotations

from collections import OrderedDict
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import math
from pathlib import Path
import secrets
import struct
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np

from tropoi.representation.archive.local_resource import resource
from tropoi.representation.archive.viewer_manifest import viewer_manifest


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def encode_frame(arrays):
    """A length-prefixed UTF-8 directory, padded to four bytes, then little-endian planes."""
    header = {}
    blocks = []
    offset = 0
    for name in ("lat", "lon", "values", "terrain", "height", "u", "v", "mask"):
        if name not in arrays:
            continue
        data = np.asarray(arrays[name], dtype="u1" if name == "mask" else "<f4")
        block = data.tobytes(order="C")
        header[name] = [offset, data.size]
        offset += len(block)
        blocks.append(block)
    directory = _json(header)
    directory += b" " * (-len(directory) % 4)
    return b"".join([struct.pack("<I", len(directory)), directory, *blocks])


def _star(manifest):
    """Geographic substellar point from the run's instantaneous illumination law."""
    physics = manifest["physics"]
    spec = physics.get("insolation") or {}
    if physics.get("enabled") is False or spec.get("kind") not in ("tidally_locked", "instantaneous"):
        return None
    times = manifest["time_axes"]["snapshots"]["values"]
    if spec["kind"] == "tidally_locked":
        point = dict(latitude=0, longitude=math.degrees(spec.get("substellar_longitude", 0)))
        positions = [point] * len(times)
    else:
        from tropoi.physics.insolation import StellarIllumination
        orbit = StellarIllumination.prescribed(**{k: v for k, v in spec.items()
                                                  if k not in ("kind", "solar_day")})
        solar_day = spec.get("solar_day", 86400)
        positions = [dict(latitude=math.degrees(orbit.declination(t)),
                          longitude=(-360 * t / solar_day + 180) % 360 - 180) for t in times]
    return dict(positions=positions, source="configured insolation geometry")


class ViewerSession:
    """Only CLI-selected roots are addressable. Frame work and the shared byte cache are serialized."""

    def __init__(self, directories, backend="cpu", cache_bytes=32 << 20):
        self.roots = [Path(path).resolve(strict=True) for path in directories]
        self.manifests = []
        for root in self.roots:
            manifest = viewer_manifest(root)
            manifest["star"] = _star(manifest)
            step_series = manifest["diagnostics"].get("timeseries")
            if step_series and step_series.get("columns"):
                # In the archive manifest this is one CSV resource. In the quantity
                # selector its scientific columns are independently selectable.
                definitions = {col["name"]: col for col in
                               (manifest.get("diagnostic_definitions") or {}).get("columns", [])}
                for column in step_series["columns"]:
                    if column in ("time_s", "dt_s", "step"):
                        continue
                    definition = definitions.get(column, {})
                    name = definition.get("meaning") or column.replace("_", " ")
                    manifest["diagnostics"]["steps." + column] = dict(
                        path=step_series["path"], kind="series", time_axis="steps", column=column,
                        presentation=dict(name=name, symbol="", units=definition.get("units"),
                                          definition=name, role="diagnostic", origin="recorded"))
                del manifest["diagnostics"]["timeseries"]
                manifest["operations"]["series"] = [
                    k for k, d in manifest["diagnostics"].items() if d["kind"] == "series"]
            self.manifests.append(manifest)
        self.backend = backend
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.bytes_used = 0
        self.lock = threading.Lock()
        self.provider = None
        self.active = None

    def frame(self, run, query):
        from tropoi.representation.visual.viewer_fields import FrameProvider

        manifest = self.manifests[run]
        field = query["field"]
        fields = manifest["fields"]
        if field not in fields or field == "wind":
            raise ValueError("Select an available scalar field")
        time = int(query["time"])
        if not 0 <= time < len(manifest["time_axes"]["snapshots"]["values"]):
            raise ValueError("Snapshot outside saved time axis")
        coordinate = query.get("coordinate", "native")
        value = float(query.get("value", 0))
        if not math.isfinite(value):
            raise ValueError("Surface value must be finite")
        if coordinate != "native" and field not in manifest["operations"].get(coordinate, []):
            raise ValueError("Coordinate is unavailable for this quantity")
        if coordinate not in ("native", "pressure", "sigma", "geopotential_height"):
            raise ValueError("Unknown surface coordinate")
        if coordinate == "native" and fields[field].get("levels"):
            levels = manifest["axes"]["vertical"]["full_levels"]
            if not value.is_integer() or not 0 <= value < len(levels):
                raise ValueError("Native level outside saved axis")
        key = (run, time, field, coordinate, value)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            if self.active != run:
                self.provider = FrameProvider(self.roots[run], manifest, backend=self.backend)
                self.active = run
            data = encode_frame(self.provider.slice(time, field, coordinate, value))
            while self.cache and self.bytes_used + len(data) > self.cache_bytes:
                self.bytes_used -= len(self.cache.popitem(last=False)[1])
            if len(data) <= self.cache_bytes:
                self.cache[key] = data
                self.bytes_used += len(data)
            return data

    def diagnostic(self, run, field):
        product = self.manifests[run]["diagnostics"].get(field)
        if not product or product["kind"] != "series":
            raise ValueError("Select an available recorded series")
        path = resource(self.roots[run], product["path"])
        if "member" in product:
            series = json.loads(path.read_text(encoding="utf-8"))
            x = series["day"][product["start_index"]:]
            y = series[product["member"]]
            return dict(x=x, y=y, x_label="Simulation day", units=product.get("units"), label=field)
        # An archive CSV can carry several step diagnostics with their own units.
        column = product.get("column")
        if column is None:
            raise ValueError("No recorded step diagnostic")
        with path.open(encoding="utf-8", newline="") as stream:
            points = [(float(row["time_s"]), float(row[column])) for row in csv.DictReader(stream)
                      if row[column] and math.isfinite(float(row[column]))]
        return dict(x=[p[0] for p in points], y=[p[1] for p in points],
                    x_label="Simulation time (s)", units=product["presentation"].get("units"), label=column)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, session, port=0):
        self.session = session
        self.token = secrets.token_urlsafe(24)
        self.assets = {p.name: p.read_bytes() for p in
                       files("tropoi.representation.visual").joinpath("viewer_assets").iterdir()
                       if p.is_file() and p.name.endswith((".html", ".js", ".css"))}
        super().__init__(("127.0.0.1", port), _Handler)
        self.authority = f"127.0.0.1:{self.server_port}"
        self.url = f"http://{self.authority}/{self.token}/0/"


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        # URLs include the private session token.
        pass

    def do_GET(self):
        server = self.server
        origin = self.headers.get("Origin")
        if (self.headers.get("Host") != server.authority or
                origin not in (None, "http://" + server.authority) or
                self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self.send_error(403)
            return
        url = urlsplit(self.path)
        parts = url.path.split("/")
        if len(parts) != 4 or parts[1] != server.token:
            self.send_error(404)
            return
        try:
            run = int(parts[2])
            session = server.session
            if not 0 <= run < len(session.roots):
                raise ValueError("Unknown capsule")
            entry = parts[3] or "index.html"
            query = parse_qs(url.query, strict_parsing=True, max_num_fields=4)
            if any(len(values) != 1 for values in query.values()):
                raise ValueError("Repeated selection parameter")
            query = {k: values[0] for k, values in query.items()}
            mime = "application/json"
            if entry == "manifest":
                data = _json(session.manifests[run])
            elif entry == "runs":
                data = _json(dict(current=run, runs=[dict(id=i, label=p.name,
                    url=f"/{server.token}/{i}/") for i, p in enumerate(session.roots)]))
            elif entry == "frame":
                data = session.frame(run, query)
                mime = "application/octet-stream"
            elif entry == "diagnostic":
                data = _json(session.diagnostic(run, query["field"]))
            elif entry in server.assets:
                data = server.assets[entry]
                mime = {".js": "text/javascript", ".html": "text/html", ".css": "text/css"}[Path(entry).suffix]
                mime += "; charset=utf-8"
            else:
                self.send_error(404)
                return
        except (ValueError, KeyError, IndexError, OSError, RuntimeError) as err:
            self._reply(400, "application/json", _json({"error": str(err)}))
            return
        self._reply(200, mime, data)

    def _reply(self, status, mime, data):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                         "object-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def serve(directories, *, port=0, open_browser=True, backend="cpu"):
    """Run in foreground so closing the command closes the private session."""
    import webbrowser

    with ViewerServer(ViewerSession(directories, backend), port) as server:
        print(f"Palintropos Viewer: {server.url}", flush=True)
        if open_browser:
            webbrowser.open(server.url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
