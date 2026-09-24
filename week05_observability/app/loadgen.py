"""Query /v1/predict continuously so the dashboard has something to show.

GENAI COMMENT:

Sends varied batch sizes randomized feature values
A small fraction of requests are missing a required feature on purpose,
to keep the error counter and the "validation" error type non-zero.
"""

import logging
import os
import random
import time

import httpx

logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("inferapi.loadgen")

JOBS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]
MARITAL = ["divorced", "married", "single", "unknown"]
EDUCATION = [
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]
CONTACT = ["cellular", "telephone"]
MONTH = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_OF_WEEK = ["mon", "tue", "wed", "thu", "fri"]
POUTCOME = ["failure", "nonexistent", "success"]


def sample_row() -> dict:
    poutcome = random.choices(POUTCOME, weights=[10, 80, 10])[0]
    previous = 0 if poutcome == "nonexistent" else random.randint(1, 5)
    pdays = 999 if poutcome == "nonexistent" else random.randint(1, 20)

    return {
        "job": random.choice(JOBS),
        "marital": random.choice(MARITAL),
        "education": random.choice(EDUCATION),
        "default": random.choices(["no", "yes", "unknown"], weights=[85, 1, 14])[0],
        "housing": random.choice(["no", "yes"]),
        "loan": random.choices(["no", "yes"], weights=[80, 20])[0],
        "contact": random.choice(CONTACT),
        "month": random.choice(MONTH),
        "day_of_week": random.choice(DAY_OF_WEEK),
        "poutcome": poutcome,
        "age": random.randint(18, 90),
        "duration": random.randint(0, 1800),
        "campaign": random.randint(1, 15),
        "pdays": pdays,
        "previous": previous,
        "emp.var.rate": round(random.uniform(-3.4, 1.4), 1),
        "cons.price.idx": round(random.uniform(92.2, 94.8), 3),
        "cons.conf.idx": round(random.uniform(-50.8, -26.9), 1),
        "euribor3m": round(random.uniform(0.6, 5.0), 3),
        "nr.employed": round(random.uniform(4963.6, 5228.1), 1),
    }


def sample_bad_row() -> dict:
    row = sample_row()
    del row["duration"]
    return row


def sample_batch() -> list[dict]:
    # GENAI COMMENT:
    #   Weighted toward small batches: real traffic is mostly single or few-row
    #   requests, with an occasional larger batch job.
    batch_size = random.choice([1, 1, 1, 2, 5, 10, 25])
    return [sample_row() for _ in range(batch_size)]


def main() -> None:
    target_url = os.environ.get("TARGET_URL", "http://inferapi:8080/v1/predict")
    interval_seconds = float(os.environ.get("REQUEST_INTERVAL_SECONDS", "0.5"))
    bad_payload_fraction = float(os.environ.get("BAD_PAYLOAD_FRACTION", "0.08"))

    logger.info("Starting load generator against %s", target_url)

    # Exercise: swap which statement wraps the other, `with httpx.Client()` around `while True`
    # instead of the reverse as it is here.
    # This order opens a new connection every iteration.
    # The other order reuses one connection across the whole loop, worth watching on the dashboard.
    while True:
        with httpx.Client(timeout=10.0) as client:
            send_bad = random.random() < bad_payload_fraction
            payload = sample_bad_row() if send_bad else sample_batch()

            try:
                response = client.post(target_url, json=payload)
                rows_sent = 1 if isinstance(payload, dict) else len(payload)
                logger.info("predict status=%s rows=%s bad=%s", response.status_code, rows_sent, send_bad)
            except httpx.HTTPError:
                logger.warning("request to %s failed", target_url, exc_info=True)

            time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
