"""One-off script: fetch real monthly rainfall from NASA POWER and write
data/rainfall_2010_2023.csv (used by project4_rainfall.ipynb).

Not run by the notebook itself (the notebook reads the committed CSV), so
Restart & Run All does not depend on network access. Re-run this script
manually to refresh the data:

    python scripts/fetch_rainfall_data.py

Source: NASA POWER (power.larc.nasa.gov), parameter PRECTOTCORR
(precipitation corrected), community AG, monthly temporal API. Values are
returned as an average daily rate (mm/day) per month; this script multiplies
by the number of days in that month to get a monthly total (mm).
"""

from __future__ import annotations

import calendar
import json
from pathlib import Path
from urllib.request import urlopen

COORDINATES = {
    "Kampala": (32.5825, 0.3476),
    "Gulu": (32.2990, 2.7746),
    "Mbarara": (30.6545, -0.6072),
}
START_YEAR = 2010
END_YEAR = 2023
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "rainfall_2010_2023.csv"


def fetch_city(city: str, lon: float, lat: float) -> dict:
    url = (
        "https://power.larc.nasa.gov/api/temporal/monthly/point"
        f"?parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}"
        f"&start={START_YEAR}&end={END_YEAR}&format=JSON"
    )
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def main() -> None:
    rows = ["region,year,month,rainfall_mm"]
    for city, (lon, lat) in COORDINATES.items():
        data = fetch_city(city, lon, lat)
        series = data["properties"]["parameter"]["PRECTOTCORR"]
        for key, daily_rate in series.items():
            year, month = int(key[:4]), int(key[4:6])
            if month == 13:  # NASA POWER's annual-average row
                continue
            days = calendar.monthrange(year, month)[1]
            monthly_total_mm = round(daily_rate * days, 2)
            rows.append(f"{city},{year},{month},{monthly_total_mm}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows) - 1} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
