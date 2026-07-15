"""Eval harness for market-data-service. Run from the service root:

    python -m eval.run_eval

Hits the real yfinance provider (keyless, but needs network) against a fixed
list of known-good and known-bad tickers, and checks that found/not-found
classification matches expectations and that returned prices are positive.

Writes eval/results.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from shared.contracts import MarketDataRequest

from client import ProviderUnavailable, YFinanceClient
from service import MarketDataService, TickerNotFound

HERE = Path(__file__).parent
RESULTS_PATH = HERE / "results.md"


def load_dataset() -> list[dict]:
    return [json.loads(line) for line in (HERE / "dataset.jsonl").read_text().splitlines() if line.strip()]


def run() -> str:
    service = MarketDataService(YFinanceClient())
    dataset = load_dataset()

    rows = []
    correct = 0
    for ex in dataset:
        try:
            resp = service.query(MarketDataRequest(ticker=ex["ticker"]))
            found, detail = True, f"price={resp.price}"
        except TickerNotFound:
            found, detail = False, "not found"
        except ProviderUnavailable as exc:
            found, detail = None, f"provider error: {exc}"

        ok = found == ex["expect_found"] and (found is False or detail.startswith("price=") and float(detail.split("=")[1]) > 0)
        correct += ok
        rows.append({"id": ex["id"], "ticker": ex["ticker"], "expect_found": ex["expect_found"], "detail": detail, "ok": ok})

    accuracy = correct / len(dataset)
    detail_rows = "\n".join(
        f"| {r['id']} | {r['ticker']} | {r['expect_found']} | {r['detail']} | {'✅' if r['ok'] else '❌'} |"
        for r in rows
    )
    report = f"""# market-data-service eval results

Live yfinance lookups against {len(dataset)} known tickers (mix of valid and invalid).

| metric | value |
|---|---|
| correctness | {correct}/{len(dataset)} = {accuracy:.2f} |

| id | ticker | expected found | detail | ok |
|---|---|---|---|---|
{detail_rows}
"""
    RESULTS_PATH.write_text(report)
    return report


def main() -> None:
    print(run())
    print(f"written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
