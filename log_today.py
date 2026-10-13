"""
log_today.py
────────────
Headless daily signal recorder for automation (GitHub Actions cron).
Loads trained artifacts, runs the exact same inference as the dashboard,
and appends the call to artifacts/signal_log.csv (deduplicated — reruns
are no-ops). Scheduled pre-market (~12:30 UTC = 08:30 ET) so each session's
call is recorded before the 09:30 open. Needs FRED_API_KEY in the env.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("log_today")

from model_engine import ModelEngine  # noqa: E402


def main() -> None:
    if not os.environ.get("FRED_API_KEY"):
        raise SystemExit("FRED_API_KEY not set in environment.")
    engine = ModelEngine().load()
    signal = engine.predict_today()
    ModelEngine.log_live_signal(signal)
    print(f"logged as_of={signal.as_of_date.date()} "
          f"signal={signal.signal} pred={signal.predicted_return_pct:+.4f}% "
          f"conf={signal.confidence_pct:.0f}% regime={signal.regime_label}")


if __name__ == "__main__":
    main()
