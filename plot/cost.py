"""The cost paragraph: the price of one resampling benchmark request.

This is the one output that reads two CSVs. It reads `resampling-runs.csv` for the
requests per second, the failure rates, and the bytes of each run, and
`resampling-requests.csv` for the S3 GET requests of each request. The second file
is optional when the prices file sets `gets_per_request`.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plot.data import (
    NoDataError,
    Output,
    as_bool,
    format_bytes,
    format_exact,
    format_short,
    select_runs,
    window_label,
)

PRICES_FILE = Path(__file__).with_name("prices.json")
"""Prices that the paragraph states."""

BYTES_PER_GB = 2**30
"""AWS bills data transfer in gigabytes of 2^30 bytes."""

SECONDS_PER_HOUR = 3_600
PRICE_KEYS = (
    "date",
    "region",
    "server_instance",
    "server_instance_usd_per_hour",
    "s3_get_usd_per_1000_requests",
    "data_out_usd_per_gb",
)
PRICE_DIGITS = 2
EVERY_WINDOW = -1
"""Window key of a GET count that applies to every window."""


@dataclass(frozen=True)
class WindowCost:
    """The price of one request at one window.

    Attributes:
        window_s: Window length in seconds.
        precomputed_gets: S3 GET requests of one precomputed request.
        precomputed_bytes: Median bytes of one precomputed request.
        precomputed_usd: Price of one precomputed request.
        server_gets: S3 GET requests of one server request.
        server_bytes: Median bytes of one server response.
        server_requests_per_second: Highest requests per second among the
            server runs with no failures, or None when every run failed.
        server_usd: Price of one server request, or None when
            `server_requests_per_second` is None.
    """

    window_s: int
    precomputed_gets: float
    precomputed_bytes: float
    precomputed_usd: float
    server_gets: float
    server_bytes: float
    server_requests_per_second: float | None
    server_usd: float | None


def load_prices(path: Path = PRICES_FILE) -> dict[str, Any]:
    """Read a prices file and check that it holds every price.

    Raises:
        ValueError: The file lacks a key of `PRICE_KEYS`.
    """
    prices: dict[str, Any] = json.loads(path.read_text())
    missing = [key for key in PRICE_KEYS if key not in prices]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    return prices


def gets_per_request(
    requests: Any | None, prices: dict[str, Any], recording: str
) -> dict[str, dict[int, float]]:
    """Return the median S3 GET requests per request, by method and window.

    The `gets_per_request` entry of the prices file, a map from method to a
    number, applies to every window and replaces the file. Otherwise the
    median of `s3_requests` over the requests of `requests` that are not
    warm-up and have status `ok` applies, per method and window.

    Raises:
        NoDataError: Neither source gives a value.
    """
    fixed = prices.get("gets_per_request")
    if fixed:
        return {
            method: {EVERY_WINDOW: float(value)}
            for method, value in fixed.items()
        }
    if requests is None:
        raise NoDataError(
            "no S3 GET counts: pass resampling-requests.csv or set "
            "gets_per_request in the prices file"
        )
    rows = requests[
        (requests["recording"] == recording)
        & ~as_bool(requests["warmup"])
        & (requests["status"] == "ok")
    ]
    medians = rows.groupby(["method", "window_s"])["s3_requests"].median()
    result: dict[str, dict[int, float]] = {}
    for (method, window), value in medians.items():
        result.setdefault(str(method), {})[int(window)] = float(value)
    return result


def _lookup(
    table: dict[str, dict[int, float]], method: str, window: int
) -> float:
    """Return the GET count of a method at a window, or of every window."""
    by_window = table.get(method, {})
    if window in by_window:
        return by_window[window]
    if EVERY_WINDOW in by_window:
        return by_window[EVERY_WINDOW]
    raise NoDataError(f"no S3 GET count of the {method} method at {window} s")


def window_costs(
    runs: Any,
    requests: Any | None,
    prices: dict[str, Any],
    recording: str = "R3",
) -> list[WindowCost]:
    """Return the price of one request of each method at each window.

    A precomputed request costs its S3 GET requests at the GET price plus its
    median bytes at the data-out price. A server request costs the hourly
    instance price divided by the requests per hour that the server sustained,
    plus its S3 GET requests at the GET price, plus its median response bytes
    at the data-out price. The sustained rate is the highest
    `requests_per_second` among the server runs of the window with
    `failure_rate` equal to 0. Both methods pay the data-out price because both
    send their bytes to a client outside AWS.

    Args:
        runs: Data frame of `resampling-runs.csv`.
        requests: Data frame of `resampling-requests.csv`, or None.
        prices: Prices from `load_prices`.
        recording: Recording to price.

    Raises:
        NoDataError: The recording has no precomputed run, or no GET count.
    """
    chosen = select_runs(runs)
    chosen = chosen[chosen["recording"] == recording]
    gets = gets_per_request(requests, prices, recording)
    get_usd = prices["s3_get_usd_per_1000_requests"] / 1000
    out_usd = prices["data_out_usd_per_gb"] / BYTES_PER_GB
    hourly = prices["server_instance_usd_per_hour"]
    costs = []
    for window in sorted(chosen["window_s"].unique()):
        at_window = chosen[chosen["window_s"] == window]
        precomputed = at_window[at_window["method"] == "precomputed"]
        if precomputed.empty:
            continue
        every_server = at_window[at_window["method"] == "server"]
        server = every_server[every_server["failure_rate"] == 0]
        precomputed_gets = _lookup(gets, "precomputed", int(window))
        server_gets = _lookup(gets, "server", int(window))
        bytes_out = float(precomputed["median_bytes"].median())
        response_bytes = float(every_server["median_bytes"].median())
        rate = (
            float(server["requests_per_second"].max()) if len(server) else None
        )
        costs.append(
            WindowCost(
                window_s=int(window),
                precomputed_gets=precomputed_gets,
                precomputed_bytes=bytes_out,
                precomputed_usd=precomputed_gets * get_usd
                + bytes_out * out_usd,
                server_gets=server_gets,
                server_bytes=response_bytes,
                server_requests_per_second=rate,
                server_usd=(
                    None
                    if rate is None
                    else hourly / (rate * SECONDS_PER_HOUR)
                    + server_gets * get_usd
                    + response_bytes * out_usd
                ),
            )
        )
    if not costs:
        raise NoDataError(f"no precomputed runs of {recording}")
    return costs


def _price(value: float) -> str:
    """Return a listed price in dollars as written in the prices file."""
    return f"${format_exact(value)}"


def _usd(value: float) -> str:
    """Return a computed price in dollars with two significant digits."""
    return f"${format_short(value, PRICE_DIGITS)}"


def cost_paragraph(
    costs: list[WindowCost], prices: dict[str, Any], recording: str
) -> str:
    """Return the cost paragraph: the prices and date, then one sentence per window."""
    lines = [
        f"Prices are {prices['region']} list prices of {prices['date']}: "
        f"{_price(prices['server_instance_usd_per_hour'])} per hour for one "
        f"{prices['server_instance']} instance, "
        f"{_price(prices['s3_get_usd_per_1000_requests'])} per 1,000 S3 GET "
        f"requests, and {_price(prices['data_out_usd_per_gb'])} per GB of "
        "data transfer out to the internet, from S3 and EC2 alike."
    ]
    for cost in costs:
        if cost.server_usd is None:
            server = "the server has no run without failures"
        else:
            server = (
                f"a server request costs {_usd(cost.server_usd)} at "
                f"{format_short(cost.server_requests_per_second or 0)} requests "
                f"per second, {format_short(cost.server_gets)} GET requests, "
                f"and a response of {format_bytes(cost.server_bytes)}"
            )
        lines.append(
            f"At the {window_label(cost.window_s)} window of {recording}, a "
            f"precomputed request makes {format_short(cost.precomputed_gets)} "
            f"GET requests and moves {format_bytes(cost.precomputed_bytes)} "
            f"for {_usd(cost.precomputed_usd)}, and {server}."
        )
    return " ".join(lines) + "\n"


def write_cost(
    runs: Any,
    requests: Any | None,
    prices: dict[str, Any],
    out_dir: Path,
    recording: str = "R3",
) -> Output:
    """Write `cost.txt` under out_dir.

    Raises:
        NoDataError: See `window_costs`.
    """
    costs = window_costs(runs, requests, prices, recording)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cost.txt"
    path.write_text(cost_paragraph(costs, prices, recording))
    return Output((path,), "")
