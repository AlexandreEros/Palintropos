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
from tropoi.representation.archive.airless import READER as AIRLESS_READER
from tropoi.representation.archive.viewer_manifest import viewer_manifest
from tropoi.representation.visual.quantities import ARCHIVED, GROUPS
from tropoi.representation.visual.selection import (
    PRESETS, SelectionError, evaluate, parse_selection, time_axis)


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def encode_frame(arrays):
    """A length-prefixed UTF-8 directory, padded to four bytes, then little-endian planes."""
    header = {}
    blocks = []
    offset = 0
    for name in ("lat", "lon", "values", "terrain", "height", "u", "v", "mask", "cells"):
        if name not in arrays:
            continue
        data = np.asarray(arrays[name], dtype="u1" if name in ("mask", "cells") else "<f4")
        block = data.tobytes(order="C")
        header[name] = [offset, data.size]
        offset += len(block)
        blocks.append(block)
    directory = _json(header)
    directory += b" " * (-len(directory) % 4)
    return b"".join([struct.pack("<I", len(directory)), directory, *blocks])


def _star(manifest):
    """The geographic substellar point, from the run's instantaneous
    illumination law.

    Current runs (``orbital``) take it from the shared geometry. Archived
    runs from before that geometry existed (``tidally_locked``,
    ``instantaneous``) are described the way those runs computed their
    forcing: a fixed point, or a uniform hour angle with the orbital
    declination."""
    physics = manifest["physics"]
    spec = physics.get("insolation") or {}
    kind = spec.get("kind")
    if physics.get("enabled") is False or kind not in ("orbital", "tidally_locked",
                                                        "instantaneous"):
        return None
    from tropoi.physics.insolation import illumination_from_spec
    if kind == "orbital":
        geometry = illumination_from_spec(spec)

        def point(t):
            lat, lon = geometry.subsolar_point(t)
            return dict(latitude=math.degrees(lat), longitude=math.degrees(lon))
    elif kind == "tidally_locked":
        fixed = dict(latitude=0, longitude=math.degrees(spec.get("substellar_longitude", 0)))

        def point(t):
            return fixed
    else:
        legacy = {k: v for k, v in spec.items() if k not in ("kind", "solar_day")}
        legacy.setdefault("period", 365.25 * 86400.0)
        geometry = illumination_from_spec(legacy)   # no spin: declination only
        solar_day = spec.get("solar_day", 86400)

        def point(t):
            return dict(latitude=math.degrees(geometry.declination(t)),
                        longitude=(-360 * t / solar_day + 180) % 360 - 180)
    # One position per saved sample of every time axis that maps can use.
    axes = {name: [point(t) for t in axis["values"]] for name, axis in manifest["time_axes"].items()
            if name in _MAP_AXES and "values" in axis}
    return dict(axes=axes, source="configured insolation geometry")


_MAP_AXES = ("snapshots", "surface_samples", "profile_samples", "every_step")


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
                                          definition=name, role="diagnostic", origin="recorded",
                                          group=ARCHIVED))
                del manifest["diagnostics"]["timeseries"]
                manifest["operations"]["series"] = [
                    k for k, d in manifest["diagnostics"].items() if d["kind"] == "series"]
            if "explorer" not in manifest:
                from tropoi.representation.visual.viewer_atmosphere import describe
                manifest["explorer"] = describe(manifest)
            # Named role assignments for the page's convenience; capability is per coordinate.
            manifest["explorer"]["presets"] = PRESETS
            # The quantity selector's groups, in the registry's order.
            manifest["explorer"]["groups"] = list(GROUPS)
            self.manifests.append(manifest)
        self.backend = backend
        self.cache_bytes = cache_bytes
        self.cache = OrderedDict()
        self.bytes_used = 0
        self.lock = threading.Lock()
        self.providers = {}

    def select(self, run, text):
        """Evaluate one selection (``selection.parse_selection``).

        Every selection is evaluated by ``selection.evaluate`` and returned as
        JSON. ``options.format = "frame"`` asks instead for the binary frame
        the globe draws: one saved state on one surface, longitude × latitude
        on the display mesh, with its terrain, heights, winds and mask.
        Returns ``(bytes, mime)``."""
        manifest = self.manifests[run]
        selection = parse_selection(manifest, text)
        key = ("select", run, json.dumps(selection, sort_keys=True))
        quantity = manifest["explorer"]["quantities"][selection["quantity"]]
        if quantity.get("recorded"):
            data = self._cached(run, key, lambda provider: _json(self.recorded(run, selection)), locked=False)
            return data, "application/json"
        frame = selection["options"]["format"] == "frame"
        if frame and not (selection["axes"] == dict(x="longitude", y="latitude")
                          and selection["time"]["role"] == "fixed"
                          and selection["vertical"]["role"] in ("fixed", "none")):
            raise SelectionError("The globe draws one saved state on one surface: longitude as x, latitude "
                                 "as y, a fixed time and a fixed vertical coordinate")

        def run_selection(provider):
            return encode_frame(provider.frame(selection)) if frame else _json(evaluate(provider, selection))

        # Frames keep the lock: the optional CUDA synthesis object is not thread-safe.
        data = self._cached(run, key, run_selection, locked=frame)
        return data, "application/octet-stream" if frame else "application/json"

    def _cached(self, run, key, evaluate, locked):
        """One byte-bounded LRU cache serves every view.

        Providers only read their archive, so evaluation runs outside the lock
        (a long zonal mean does not block map requests) unless ``locked``.
        """
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            provider = self.providers.get(run)
            if provider is None:
                provider = self.providers[run] = provider_for(self.roots[run], self.manifests[run],
                                                              backend=self.backend)
            if locked:
                data = evaluate(provider)
        if not locked:
            data = evaluate(provider)
        with self.lock:
            if key not in self.cache:
                while self.cache and self.bytes_used + len(data) > self.cache_bytes:
                    self.bytes_used -= len(self.cache.popitem(last=False)[1])
                if len(data) <= self.cache_bytes:
                    self.cache[key] = data
                    self.bytes_used += len(data)
            return data

    def recorded(self, run, selection):
        """A recorded series, at its own cadence, as a selection payload."""
        manifest = self.manifests[run]
        quantity = selection["quantity"]
        field = manifest["explorer"]["quantities"][quantity]["recorded"]
        product = manifest["diagnostics"].get(field)
        if not product or product["kind"] != "series":
            raise ValueError("Select an available recorded series")
        path = resource(self.roots[run], product["path"])
        if "member" in product:
            series = json.loads(path.read_text(encoding="utf-8"))
            x = [86400.0 * day for day in series["day"][product["start_index"]:]]
            y = series[product["member"]]
            return _series(quantity, product, x, y)
        # An archive CSV can carry several step diagnostics with their own units.
        column = product.get("column")
        if column is None:
            raise ValueError("No recorded step diagnostic")
        with path.open(encoding="utf-8", newline="") as stream:
            points = [(float(row["time_s"]), float(row[column])) for row in csv.DictReader(stream)
                      if row[column] and math.isfinite(float(row[column]))]
        return _series(quantity, product, [p[0] for p in points], [p[1] for p in points])


def _series(quantity, product, x, y):
    if len(x) != len(y):
        raise ValueError("Recorded time and value counts disagree")
    none = dict(role="none")
    return dict(quantity=quantity, presentation=product["presentation"],
                roles=dict(longitude="none", latitude="none", vertical="none", time="x"),
                axes=dict(x="time", y=None), description="Recorded series",
                coordinates=dict(longitude=none, latitude=none, vertical=none,
                                 time=dict(role="x", axis="recorded", count=len(x))),
                x=time_axis(x, label="Recorded time"), values=y, companions=None,
                sampling=f"recorded in {product['path']} at the run's own cadence", notes=[])


def provider_for(root, manifest, *, backend="cpu"):
    """The evaluator of an archive, chosen by the reader that described it."""
    if manifest["source"].get("reader") == AIRLESS_READER:
        from tropoi.representation.visual.viewer_columns import ColumnArchiveProvider
        return ColumnArchiveProvider(root, manifest, backend=backend)
    from tropoi.representation.visual.viewer_fields import FrameProvider
    return FrameProvider(root, manifest, backend=backend)


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
            # The page carries its own state (time, cursor, preset); a selection
            # request carries exactly one parameter, q.
            query = parse_qs(url.query, strict_parsing=True, max_num_fields=6) if url.query else {}
            if any(len(values) != 1 for values in query.values()):
                raise ValueError("Repeated query parameter")
            if entry == "select" and set(query) != {"q"}:
                raise ValueError("The only query parameter of a selection is q")
            mime = "application/json"
            if entry == "manifest":
                data = _json(session.manifests[run])
            elif entry == "runs":
                data = _json(dict(current=run, runs=[dict(id=i, label=p.name,
                    url=f"/{server.token}/{i}/") for i, p in enumerate(session.roots)]))
            elif entry == "select":
                data, mime = session.select(run, query["q"][0])
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
