"""Generate a synthetic opportunities dataset with the same 13 columns the
Salesforce connector produces (after cleaning).

Everything here is invented: names, owners, dates and figures. The structure
mirrors the Opportunity export the chatbot was originally built on.

Usage:
    python generate_sample_data.py            # writes data/opportunities.csv
    python generate_sample_data.py --rows 300 # more rows
"""
import argparse
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 42

# account -> (code prefixes, share of rows)
ACCOUNTS = {
    "BMW": ["BMW"],
    "Mercedes-Benz": ["MBV", "MBW"],
    "Audi": ["AU", "MBV"],
    "Volkswagen": ["VW"],
    "Land Rover": ["LRL"],
    "Aston Martin": ["AM"],
    "Porsche": ["PO"],
    "Bentley": ["BE"],
}

# Fake people (not real employees)
OWNERS = ["Omar Tovalin", "Lena Hartmann", "Karim Belhadj", "Sofia Marchetti",
          "Jonas Weber", "Amira Ben Salah"]

# (stage, weight)
STAGES = [("Draft Phase", 70), ("Qualification", 10), ("Negotiation", 8),
          ("Closed Won", 8), ("Closed Lost", 4)]

TODAY = date(2026, 10, 20)


def fmt_date(d: date) -> str:
    return d.strftime("%d.%m.%Y")


def fmt_dt(d: datetime) -> str:
    return d.strftime("%d.%m.%Y, %H:%M")


def make_name(prefix: str, rng: random.Random) -> str:
    style = rng.choice(["plain", "plain", "slash", "dual"])
    num = rng.randint(100, 999)
    if prefix == "AM":
        return f"AM{rng.randint(1, 60):03d}__"
    if style == "slash":
        return f"{prefix}{num}/{rng.randint(0, 9)}__"
    if style == "dual":
        return f"{prefix}{num}/X{rng.randint(100, 999)}__"
    return f"{prefix}{num}__"


def generate(rows: int, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    names = set()
    records = []
    accounts = list(ACCOUNTS)

    while len(records) < rows:
        account = accounts[len(records) % len(accounts)] if len(records) < 40 else rng.choice(accounts)
        prefix = rng.choice(ACCOUNTS[account])
        name = make_name(prefix, rng)
        if name in names:
            continue
        names.add(name)

        sop = date(rng.randint(2025, 2031), rng.randint(1, 12), 1)
        eop = date(sop.year + rng.randint(5, 10), rng.randint(1, 12), 1)
        dos = sop - timedelta(days=rng.randint(300, 900))

        volume = rng.choice([0, rng.randint(5_000, 60_000), rng.randint(60_000, 600_000),
                             rng.randint(600_000, 1_500_000)])
        unit_price = rng.uniform(80, 900)
        sales = round(volume * unit_price, 2)

        created = datetime(2025, 6, 19, 17, rng.randint(0, 5))
        modified = created + timedelta(minutes=rng.randint(0, 3)) \
            if rng.random() < 0.8 else created + timedelta(days=rng.randint(1, 200), hours=rng.randint(0, 8))

        stage = rng.choices([s for s, _ in STAGES], weights=[w for _, w in STAGES])[0]

        records.append({
            "Opportunity Name": name,
            "Stage": stage,
            "Actual Priority": str(rng.choice([1, 2, 3])),
            "SOP": fmt_date(sop),
            "EOP": fmt_date(eop),
            "DOS": fmt_date(dos),
            "Created Date": fmt_dt(created),
            "Last Modified Date": fmt_dt(min(modified, datetime.combine(TODAY, datetime.min.time()))),
            "Total Sales": sales,
            "Total Volume": float(volume),
            "Predecessor Carline": "",
            "Account Name": account,
            "Owner Full Name": rng.choice(OWNERS),
        })

    df = pd.DataFrame(records)

    # Guarantee a few known, easy-to-demo opportunities.
    demo = {
        0: ("BMWGEN6__", "BMW", "Draft Phase"),
        1: ("AM003__", "Aston Martin", "Closed Won"),
        2: ("VW426__", "Volkswagen", "Draft Phase"),
    }
    for idx, (n, acc, stage) in demo.items():
        df.loc[idx, ["Opportunity Name", "Account Name", "Stage"]] = [n, acc, stage]
    df = df.drop_duplicates(subset="Opportunity Name", keep="first").reset_index(drop=True)

    # Link ~35% of rows to an earlier carline of the same account (predecessor).
    for i in range(len(df)):
        if rng.random() < 0.35:
            same = df[(df["Account Name"] == df.loc[i, "Account Name"]) & (df.index != i)]
            if not same.empty:
                df.loc[i, "Predecessor Carline"] = same.sample(1, random_state=rng.randint(0, 10**6))["Opportunity Name"].iloc[0]

    # Real-world messiness: a missing DOS and a zero-value opportunity.
    df.loc[df.index[-1], "DOS"] = None
    df.loc[df.index[-2], ["Total Sales", "Total Volume"]] = [0.0, 0.0]

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=120)
    parser.add_argument("--out", default="data/opportunities.csv")
    args = parser.parse_args()

    df = generate(args.rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
