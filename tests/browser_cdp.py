"""A minimal, dependency-free headless-Chrome driver for offline page checks.

Opens standalone HTML files from ``file://`` URLs in a headless Chromium
(Chrome, Chromium or Edge) and drives them over the DevTools protocol, using
only the standard library (a small WebSocket client is included). Networking
is blocked three ways: a dead proxy, a host resolver that maps every name to
NOTFOUND, and DevTools offline emulation. Every request the page issues, every
console error and every uncaught exception is recorded.

WebGL is software-rendered (SwiftShader), so a check passes on machines with
no GPU. ``find_browser()`` returns ``None`` when no browser is installed; tests
skip in that case. Set ``TROPOI_BROWSER`` to a browser executable to choose one.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request

_CANDIDATES = ("google-chrome", "google-chrome-stable", "chromium",
               "chromium-browser", "chrome", "msedge", "microsoft-edge")
_WINDOWS = (r"Google\Chrome\Application\chrome.exe",
            r"Chromium\Application\chrome.exe",
            r"Microsoft\Edge\Application\msedge.exe")


def find_browser() -> str | None:
    """A headless-capable Chromium executable, or ``None``."""
    env = os.environ.get("TROPOI_BROWSER")
    if env:
        return env if pathlib.Path(env).exists() else None
    for name in _CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    for base in (os.environ.get("PROGRAMFILES"),
                 os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")):
        for rel in _WINDOWS:
            if base and (pathlib.Path(base) / rel).exists():
                return str(pathlib.Path(base) / rel)
    mac = pathlib.Path("/Applications/Google Chrome.app/Contents/MacOS/"
                       "Google Chrome")
    return str(mac) if mac.exists() else None


class _WebSocket:
    """Client side of RFC 6455, text frames only (enough for DevTools)."""

    def __init__(self, url: str, timeout: float):
        u = urllib.parse.urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port),
                                             timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\n"
                           "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("DevTools closed the handshake")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError(head.decode(errors="replace"))
        self.buf = head.split(b"\r\n\r\n", 1)[1]

    def _read(self, n: int) -> bytes:
        while len(self.buf) < n:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("DevTools connection closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, text: str) -> None:
        data = text.encode()
        n = len(data)
        head = bytes([0x81])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        body = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        self.sock.sendall(head + mask + body)

    def recv(self) -> str:
        parts = []
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            payload = self._read(n)
            opcode = b0 & 0x0F
            if opcode == 0x8:
                raise ConnectionError("DevTools closed the connection")
            if opcode == 0x9:                     # ping -> pong
                self.sock.sendall(bytes([0x8A, 0x80 | len(payload)])
                                  + b"\0\0\0\0" + payload)
                continue
            if opcode in (0x1, 0x0):
                parts.append(payload)
                if b0 & 0x80:
                    return b"".join(parts).decode()

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class OfflinePage:
    """One headless browser with one page, networking blocked.

    Use as a context manager. ``open(path)`` loads a local file; afterwards
    ``requests``, ``console_errors`` and ``exceptions`` hold what the page did.
    """

    def __init__(self, browser: str, *, width: int = 1280, height: int = 1000,
                 timeout: float = 60.0):
        self.timeout = timeout
        self.profile = tempfile.mkdtemp(prefix="tropoi-cdp-")
        self.proc = subprocess.Popen(
            [browser, "--headless=new", "--remote-debugging-port=0",
             f"--user-data-dir={self.profile}", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions",
             "--disable-background-networking", "--disable-sync",
             "--disable-component-update", "--no-sandbox",
             "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
             "--proxy-server=127.0.0.1:9", "--proxy-bypass-list=<-loopback>",
             "--host-resolver-rules=MAP * ~NOTFOUND",
             f"--window-size={width},{height}", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws = None
        self.version = None
        self._id = 0
        self.events: list[dict] = []
        self.requests: list[str] = []
        self.console_errors: list[str] = []
        self.exceptions: list[str] = []

    # -- lifecycle --------------------------------------------------------
    def __enter__(self):
        port_file = pathlib.Path(self.profile) / "DevToolsActivePort"
        deadline = time.monotonic() + self.timeout
        while not (port_file.exists() and port_file.read_text().strip()):
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise RuntimeError("the browser did not start DevTools")
            time.sleep(0.05)
        port = int(port_file.read_text().split()[0])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        base = f"http://127.0.0.1:{port}"
        self.version = json.loads(opener.open(f"{base}/json/version",
                                              timeout=10).read())["Browser"]
        targets = json.loads(opener.open(f"{base}/json/list",
                                         timeout=10).read())
        page = next(t for t in targets if t["type"] == "page")
        self.ws = _WebSocket(page["webSocketDebuggerUrl"], self.timeout)
        for domain in ("Page", "Runtime", "Log", "Network"):
            self.call(f"{domain}.enable")
        self.call("Network.emulateNetworkConditions", offline=True, latency=0,
                  downloadThroughput=-1, uploadThroughput=-1)
        return self

    def __exit__(self, *exc):
        if self.ws:
            self.ws.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        shutil.rmtree(self.profile, ignore_errors=True)

    # -- protocol ---------------------------------------------------------
    def _record(self, msg: dict) -> None:
        method, p = msg.get("method"), msg.get("params", {})
        if method == "Network.requestWillBeSent":
            self.requests.append(p["request"]["url"])
        elif method == "Runtime.exceptionThrown":
            d = p["exceptionDetails"]
            self.exceptions.append(d.get("exception", {}).get(
                "description", d.get("text", "")))
        elif method == "Runtime.consoleAPICalled" and p["type"] == "error":
            self.console_errors.append(" ".join(
                str(a.get("value", a.get("description", "")))
                for a in p["args"]))
        elif method == "Log.entryAdded" and p["entry"]["level"] == "error":
            self.console_errors.append(p["entry"]["text"])
        if method:
            self.events.append(msg)

    def call(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method,
                                 "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            self._record(msg)

    def eval(self, expression: str):
        """The JSON value of ``expression`` (promises awaited)."""
        r = self.call("Runtime.evaluate", expression=expression,
                      awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(d.get("exception", {}).get(
                "description", d.get("text")))
        return r["result"].get("value")

    def open(self, path: pathlib.Path) -> str:
        url = pathlib.Path(path).resolve().as_uri()
        self.events.clear()
        self.call("Page.navigate", url=url)
        deadline = time.monotonic() + self.timeout
        while not any(e.get("method") == "Page.loadEventFired"
                      for e in self.events):
            if time.monotonic() > deadline:
                raise TimeoutError(f"{url} did not load")
            self._record(json.loads(self.ws.recv()))
        return url

    def wait_for(self, expression: str, timeout: float = 20.0):
        """Poll ``expression`` until truthy; return its value."""
        deadline = time.monotonic() + timeout
        while True:
            value = self.eval(expression)
            if value:
                return value
            if time.monotonic() > deadline:
                raise TimeoutError(expression)
            time.sleep(0.05)

    def screenshot(self, path: pathlib.Path) -> pathlib.Path:
        data = self.call("Page.captureScreenshot", format="png")["data"]
        pathlib.Path(path).write_bytes(base64.b64decode(data))
        return pathlib.Path(path)
