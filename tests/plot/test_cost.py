import json

import pytest

from plot.cost import (
    BYTES_PER_GB,
    cost_paragraph,
    gets_per_request,
    load_prices,
    window_costs,
    write_cost,
)
from plot.data import NoDataError

PRICES = {
    "date": "2026-10-26",
    "region": "us-east-1",
    "server_instance": "c6i.8xlarge",
    "server_instance_usd_per_hour": 3.6,
    "s3_get_usd_per_1000_requests": 1.0,
    "data_out_usd_per_gb": 0.5,
    "gets_per_request": None,
}


def test_shipped_prices_file_holds_every_key_and_a_date():
    prices = load_prices()
    assert prices["region"] == "us-east-1"
    assert len(prices["date"]) == 10
    assert prices["check_on_run_date"]


def test_load_prices_rejects_a_file_with_a_missing_key(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"date": "2026-10-26"}))
    with pytest.raises(ValueError, match="missing keys"):
        load_prices(path)


def test_gets_per_request_is_the_median_of_ok_requests_after_warm_up(requests):
    gets = gets_per_request(requests, PRICES, "R3")
    assert gets["server"] == {10: 4.0, 600: 4.0}
    assert gets["precomputed"] == {10: 3.0, 600: 3.0}


def test_gets_per_request_input_replaces_the_requests_file(requests):
    prices = PRICES | {"gets_per_request": {"server": 9, "precomputed": 7}}
    gets = gets_per_request(requests, prices, "R3")
    assert gets["server"] == {-1: 9.0}
    assert gets_per_request(None, prices, "R3")["precomputed"] == {-1: 7.0}


def test_gets_per_request_raises_without_any_source():
    with pytest.raises(NoDataError, match="gets_per_request"):
        gets_per_request(None, PRICES, "R3")


def test_precomputed_price_is_gets_plus_bytes_out(runs, requests):
    costs = {c.window_s: c for c in window_costs(runs, requests, PRICES)}
    cost = costs[10]
    assert cost.precomputed_gets == 3.0
    assert cost.precomputed_bytes == 4_000_010
    expected = 3 * 0.001 + 4_000_010 * 0.5 / BYTES_PER_GB
    assert cost.precomputed_usd == pytest.approx(expected)


def test_server_price_uses_the_best_run_without_failures(runs, requests):
    costs = {c.window_s: c for c in window_costs(runs, requests, PRICES)}
    # At 10 s the runs without failures are 1, 10, and the repeated 100 users.
    # Their rates are 1.1, 3.26, and 5.0 requests per second.
    cost = costs[10]
    assert cost.server_requests_per_second == pytest.approx(5.0)
    response = 1000 * 0.5 / BYTES_PER_GB
    assert cost.server_bytes == 1000
    assert cost.server_usd == pytest.approx(
        3.6 / (5.0 * 3600) + 4 * 0.001 + response
    )
    # At 10 min the runs at 100 and 1000 users failed. 10 users sustained
    # sqrt(10) + 6 requests per second.
    other = costs[600]
    assert other.server_requests_per_second == pytest.approx(10**0.5 + 6)
    assert other.server_usd == pytest.approx(
        3.6 / ((10**0.5 + 6) * 3600) + 4 * 0.001 + response
    )


def test_server_price_ignores_a_run_that_was_not_run(runs, requests):
    changed = runs.copy()
    mask = (
        (changed["method"] == "server")
        & (changed["window_s"] == 10)
        & (changed["repeated"] == "true")
    )
    changed.loc[mask, "not_run"] = "true"
    costs = {c.window_s: c for c in window_costs(changed, requests, PRICES)}
    # The 5.0 requests per second of the repeated run no longer counts.
    assert costs[10].server_requests_per_second == pytest.approx(10**0.5 + 0.1)


def test_server_price_is_none_when_every_run_failed(runs, requests):
    failed = runs.copy()
    failed.loc[failed["method"] == "server", "failure_rate"] = 1.0
    costs = window_costs(failed, requests, PRICES)
    assert all(c.server_usd is None for c in costs)
    assert "no run without failures" in cost_paragraph(costs, PRICES, "R3")


def test_window_costs_raise_without_precomputed_runs(runs, requests):
    with pytest.raises(NoDataError):
        window_costs(runs[runs["method"] == "server"], requests, PRICES)


def test_paragraph_states_prices_and_date(runs, requests):
    text = cost_paragraph(window_costs(runs, requests, PRICES), PRICES, "R3")
    assert text.startswith(
        "Prices are us-east-1 list prices of 2026-10-26: $3.6 per hour for "
        "one c6i.8xlarge instance, $1.0 per 1,000 S3 GET requests, and $0.5 "
        "per GB of data transfer out to the internet, from S3 and EC2 alike."
    )
    assert "At the 10 s window of R3, a precomputed request makes 3 GET" in text
    assert "At the 10 min window of R3" in text
    assert "GET requests, and a response of" in text
    assert text.isascii()


def test_write_cost_writes_cost_txt(runs, requests, tmp_path):
    output = write_cost(runs, requests, PRICES, tmp_path)
    assert [p.name for p in output.paths] == ["cost.txt"]
    assert output.paths[0].read_text().startswith("Prices are")
