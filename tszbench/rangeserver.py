"""Local HTTP/1.1 file server with byte ranges, for tests and the smoke run.

It stands in for the nginx file server. It answers HEAD and GET with a single
byte range, keeps connections alive, and lists a directory as HTML links.
"""

import html
import re
import socket
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")
"""A `Range` header of one byte range. Any other `Range` reads the whole file."""

SHUTDOWN_TIMEOUT_S = 5.0
"""Seconds `RunningServer.stop` waits for the serving thread."""


class RangeRequestHandler(BaseHTTPRequestHandler):
    """Serve the files below `directory` with HEAD and single byte ranges."""

    protocol_version = "HTTP/1.1"
    directory: Path

    def log_message(self, format: str, *args: object) -> None:
        """Write no access log."""

    def do_HEAD(self) -> None:  # noqa: N802
        """Answer a HEAD request."""
        self.respond(head_only=True)

    def do_GET(self) -> None:  # noqa: N802
        """Answer a GET request."""
        self.respond(head_only=False)

    def resolve(self) -> Path | None:
        """Return the file the request names, or None outside `directory`."""
        relative = unquote(urlsplit(self.path).path).lstrip("/")
        root = self.directory.resolve()
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            return None
        return target

    def send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        head_only: bool,
        headers: dict[str, str] | None = None,
        length: int | None = None,
    ) -> None:
        """Send a response whose body is body, or no body for a HEAD.

        Args:
            status: HTTP status code.
            body: Payload of a GET.
            content_type: Value of `Content-Type`.
            head_only: True to send the headers alone.
            headers: Further headers.
            length: Value of `Content-Length` when it is not the body size.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header(
            "Content-Length", str(len(body) if length is None else length)
        )
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def respond(self, head_only: bool) -> None:
        """Answer with the file, a byte range of it, or a directory listing."""
        target = self.resolve()
        if target is None or not target.exists():
            self.send_bytes(404, b"", "text/plain", head_only)
            return
        if target.is_dir():
            self.send_listing(target, head_only)
            return
        size = target.stat().st_size
        header = self.headers.get("Range")
        match = RANGE_PATTERN.match(header) if header else None
        if match is None:
            body = b"" if head_only else target.read_bytes()
            self.send_bytes(
                200,
                body,
                "application/octet-stream",
                head_only,
                {"Accept-Ranges": "bytes"},
                length=size,
            )
            return
        first, last = match.groups()
        if first == "":
            if last == "" or int(last) == 0:
                self.send_range_error(size, head_only)
                return
            start = max(0, size - int(last))
            stop = size
        else:
            start = int(first)
            stop = size if last == "" else min(int(last) + 1, size)
        if start >= size or stop <= start:
            self.send_range_error(size, head_only)
            return
        with target.open("rb") as handle:
            handle.seek(start)
            body = handle.read(stop - start)
        self.send_bytes(
            206,
            body,
            "application/octet-stream",
            head_only,
            {
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{stop - 1}/{size}",
            },
        )

    def send_listing(self, target: Path, head_only: bool) -> None:
        """Send the entries of a directory as links, after a slash redirect."""
        path = urlsplit(self.path).path
        if not path.endswith("/"):
            self.send_response(301)
            self.send_header("Location", path + "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        names = sorted(
            child.name + ("/" if child.is_dir() else "")
            for child in target.iterdir()
        )
        links = "\n".join(
            f'<a href="{quote(name)}">{html.escape(name)}</a><br>'
            for name in names
        )
        page = f"<html><body>\n{links}\n</body></html>\n".encode()
        self.send_bytes(200, page, "text/html", head_only)

    def send_range_error(self, size: int, head_only: bool) -> None:
        """Send 416 for a range outside a file of size bytes."""
        self.send_bytes(
            416,
            b"",
            "text/plain",
            head_only,
            {"Content-Range": f"bytes */{size}"},
        )


class FileServer(ThreadingHTTPServer):
    """Threaded HTTP/1.1 server with a deep accept queue."""

    daemon_threads = True
    request_queue_size = 256


@dataclass(frozen=True)
class RunningServer:
    """A file server that serves on a thread of this process.

    Attributes:
        server: The server.
        thread: The thread that runs `serve_forever`.
        url: Base URL of the served directory, ending in `/`.
    """

    server: FileServer
    thread: threading.Thread
    url: str

    def stop(self) -> None:
        """Stop serving and close the socket."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=SHUTDOWN_TIMEOUT_S)


def serve_directory(
    directory: Path, host: str = "127.0.0.1", port: int = 0
) -> RunningServer:
    """Serve directory on a daemon thread and return the running server.

    Args:
        directory: Root of the served files.
        host: Address to listen on.
        port: Port to listen on, or 0 for a free port.
    """
    handler = type(
        "BoundHandler", (RangeRequestHandler,), {"directory": Path(directory)}
    )
    server = FileServer((host, port), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    return RunningServer(
        server, thread, f"http://{host}:{server.server_address[1]}/"
    )


def free_port(host: str = "127.0.0.1") -> int:
    """Return a TCP port on host that no socket holds at the time of the call."""
    with socket.socket() as probe:
        probe.bind((host, 0))
        return int(probe.getsockname()[1])
