"""Salesforce connector + data loader.

`load_data()` is the single entry point used by the app:
  * If SF_USERNAME / SF_PASSWORD / SF_TOKEN are set, it queries the Opportunity
    object in Salesforce (the original behaviour).
  * Otherwise it falls back to the local CSV (data/opportunities.csv), which has
    exactly the same columns as the cleaned Salesforce DataFrame.

Credentials are never stored in code. Put them in a `.env` file (git-ignored)
or export them as environment variables. See `.env.example`.
"""
import os
from pathlib import Path

import pandas as pd

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_CSV = Path(__file__).parent / "data" / "opportunities.csv"

# Where the data of the last load_data() call came from: "salesforce" or "csv".
LAST_SOURCE = None

# Columns the chatbot expects after cleaning.
EXPECTED_COLUMNS = [
    "Opportunity Name", "Stage", "Actual Priority", "SOP", "EOP", "DOS",
    "Created Date", "Last Modified Date", "Total Sales", "Total Volume",
    "Predecessor Carline", "Account Name", "Owner Full Name",
]

SOQL_QUERY = """
    SELECT Name, Account.Name, Owner.Name, StageName,
           Actual_Priority__c, SOP__c, EOP__c, Expected_DOS__c,
           CreatedDate, LastModifiedDate, Total_Sales__c,
           Total_Volume__c, Predecessor_Carline__c
    FROM Opportunity
    WHERE Name != null
"""


# ---------------------------------------------------------------------------
# Salesforce
# ---------------------------------------------------------------------------

def salesforce_configured():
    return all(os.environ.get(k) for k in ("SF_USERNAME", "SF_PASSWORD", "SF_TOKEN"))


def get_salesforce_connection():
    """Create and return a Salesforce connection from environment variables.

    SF_DOMAIN: 'test' for a sandbox (the original setup), 'login' for a
    production org or a free Developer Edition org.
    """
    from simple_salesforce import Salesforce  # imported lazily: CSV mode needs no Salesforce package

    missing = [k for k in ("SF_USERNAME", "SF_PASSWORD", "SF_TOKEN") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    return Salesforce(
        username=os.environ["SF_USERNAME"],
        password=os.environ["SF_PASSWORD"],
        security_token=os.environ["SF_TOKEN"],
        domain=os.environ.get("SF_DOMAIN", "test"),
    )


def fetch_opportunities(sf):
    """Query Salesforce and return a cleaned DataFrame of opportunities."""
    results = sf.query_all(SOQL_QUERY)
    df = pd.DataFrame(results["records"])

    if df.empty:
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if "attributes" in df.columns:  # Salesforce metadata column
        df = df.drop("attributes", axis=1)

    df = _extract_nested_fields(df)
    df = _rename_columns(df)
    df = _format_dates(df)
    return df


def _extract_nested_fields(df):
    """Flatten nested Salesforce objects like Account.Name and Owner.Name."""
    def extract(series):
        return series.apply(lambda x: x.get("Name") if isinstance(x, dict) else x)

    if "Account" in df.columns:
        df["Account_Name"] = extract(df["Account"])
        df = df.drop("Account", axis=1)

    if "Owner" in df.columns:
        df["Owner_Name"] = extract(df["Owner"])
        df = df.drop("Owner", axis=1)

    return df


def _rename_columns(df):
    column_mapping = {
        "Name": "Opportunity Name",
        "Account_Name": "Account Name",
        "Owner_Name": "Owner Full Name",
        "StageName": "Stage",
        "Actual_Priority__c": "Actual Priority",
        "SOP__c": "SOP",
        "EOP__c": "EOP",
        "Expected_DOS__c": "DOS",
        "CreatedDate": "Created Date",
        "LastModifiedDate": "Last Modified Date",
        "Total_Sales__c": "Total Sales",
        "Total_Volume__c": "Total Volume",
        "Predecessor_Carline__c": "Predecessor Carline",
    }
    df = df.rename(columns=column_mapping)
    df.columns = df.columns.str.strip()
    return df


def _format_dates(df):
    """Format date columns as dd.mm.YYYY (and timestamps in Europe/Paris time)."""
    for col in ["SOP", "EOP", "DOS"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            df[col] = parsed.dt.strftime("%d.%m.%Y")

    for col in ["Created Date", "Last Modified Date"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = parsed.dt.tz_convert("Europe/Paris").dt.strftime("%d.%m.%Y, %H:%M")

    return df


# ---------------------------------------------------------------------------
# CSV fallback + single entry point
# ---------------------------------------------------------------------------

def load_csv(path=None):
    """Load the local dataset (same columns as the cleaned Salesforce data)."""
    path = Path(path or os.environ.get("DATA_CSV", DEFAULT_CSV))
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python generate_sample_data.py` to create it."
        )
    df = pd.read_csv(path, dtype={"Actual Priority": str}, keep_default_na=True)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df


def load_data(source=None):
    """Return the opportunities DataFrame.

    source: 'salesforce', 'csv', or None (auto: Salesforce if credentials are
    set, otherwise CSV). The DATA_SOURCE environment variable does the same.
    """
    global LAST_SOURCE
    source = (source or os.environ.get("DATA_SOURCE") or "auto").lower()

    if source == "salesforce" or (source == "auto" and salesforce_configured()):
        df = fetch_opportunities(get_salesforce_connection())
        LAST_SOURCE = "salesforce"
        print(f"[data] Loaded {len(df)} opportunities from Salesforce")
        return df

    df = load_csv()
    LAST_SOURCE = "csv"
    print(f"[data] Loaded {len(df)} opportunities from CSV")
    return df
