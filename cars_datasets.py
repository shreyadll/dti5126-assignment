import math
import pandas as pd
import numpy as np
import re
from io import StringIO
import matplotlib.pyplot as plt
from collections import Counter

path_to_dataset = "Cars-Datasets-2025-2.csv"

# Below are some of the regular expression to match numbers and alpha-numeric values
# Number with optional sign, value separators, and decimal part.
NUM_RE = r"[-+]?\d[\d,]*\.?\d*"
# To match range like low - high
RANGE_RE = re.compile(rf"{NUM_RE}\s*-\s*{NUM_RE}")
# To match a string that only contains number
ONLY_NUM_RE = re.compile(rf"^\s*{NUM_RE}\s*$")

# This function helps to read the csv file of the dataset
# The data might have odd encodings, so we use a best fit technique for it
def read_csv_robust(path, **kwargs):
    tries = ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]
    last_err = None
    for enc in tries:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError as e:
            last_err = e
    return pd.read_csv(path, encoding="ISO-8859-1", low_memory=False, **kwargs)

extra_na = ["", " ", "n/a", "na", "N/A", "NA", "null", "NULL", "none", "None",
            "-", "--", "—", "tbd", "TBD", "unknown", "Unknown", "?", "N.A.", "Not Available"]

# Helper function to print percentage consistently
def pct(x, total):
    return f"{(100.0*x/total):.1f}%"

# We want to inspect certain column values to understand the data pattern
# Inspection include knowing the count of:
# (a) non-nulls (b) values with commas (c) range like values
# (d) pure numbers (e) unit token hits such as cc, kWh, Nm
def inspect_patterns(s, unit_tokens=None, name=""):
    total, nn = len(s), s.notna().sum()
    vals = s.dropna().astype(str).str.strip()
    with_commas = int(vals.str.contains(",").sum())
    with_range  = int(vals.str.contains(RANGE_RE).sum())
    only_num    = int(vals.str.match(ONLY_NUM_RE).sum())
    print(f"\n[Inspecting the ] {name} Column")
    print(f"  - Non-null: {nn}/{total}")
    print(f"  - Contains commas: {with_commas} ({pct(with_commas, nn)})")
    print(f"  - Contains range (a-b): {with_range} ({pct(with_range, nn)})")
    print(f"  - Pure numeric only: {only_num} ({pct(only_num, nn)})")
    if unit_tokens:
        counts = {u:int(vals.str.contains(fr'\b{u}\b', case=True).sum()) for u in unit_tokens}
        print("  - Unit tokens: " + ", ".join([f"{k}:{v} ({pct(v, nn)})" for k,v in counts.items()]))

# We want to polish the price values. There are certain prices that are not
# properly written. So, we will re-write them without any alteration.
def _normalize_price_text(s):
    if pd.isna(s):
        return s
    s = str(s)
    # normalize the spaces values
    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2007", " ")
    # fix euro symbol
    s = s.replace("â‚¬", "€").replace("\x80", "€")
    # normalize dashes to a simple hyphen
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\x96", "-")
    # collapse spaces if any
    s = re.sub(r"\s+", " ", s).strip()
    return s

# This will return a min-max range if the price value is a range, otherwise it
# will return a single number as a string. We will also strip the symbol such as
# $ from the price. The price is returned as number without any commas, and non
# price texts are returned as NaN
def price_keep_range(val):
    if pd.isna(val):
        return np.nan
    s = _normalize_price_text(val)

    # treat obvious non-price rows as missing
    if re.search(r"(?i)\b(n/?a|concept)\b", s):
        return np.nan

    # drop currency tokens so we keep numbers only
    s_nocurr = re.sub(r"(?i)(\$|€|EUR)", "", s)

    # extract numbers
    nums = re.findall(NUM_RE, s_nocurr)
    nums = [re.sub(",", "", n).strip() for n in nums if n.strip() != ""]
    if not nums:
        return np.nan

    # keep ranges exactly like in the dataset example
    return f"{nums[0]} - {nums[-1]}" if len(nums) >= 2 else nums[0]

# We want to add a new column beside the original "Cars Prices" column.
# All future operations will be computed using this new column
def add_cars_price_amount_column(df):
    anchor = "Cars Prices"
    df["cars_price_amount"] = df[anchor].map(price_keep_range)

    # Move the new column to sit immediately after the anchor column
    cols = df.columns.tolist()
    if "cars_price_amount" in cols:
        cols.remove("cars_price_amount")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "cars_price_amount")
    return df[cols]

# Simple normalization for the column values
def _cap_words_keep_hyphen(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())

# We want to keep common acronyms in uppercase
def _fix_acronyms(s: str) -> str:
    s = re.sub(r"\bEv\b", "EV", s)
    s = re.sub(r"\bCng\b", "CNG", s)
    s = re.sub(r"\bAwd\b", "AWD", s)
    return s

CANON_MAP = {
    "plug in hyrbrid": "Plug-in Hybrid",
    "plug-in hybrid": "Plug-in Hybrid",
    "plug in hybrid": "Plug-in Hybrid",
    "diesel hybrid": "Diesel Hybrid",
    "ev": "EV",
    "cng": "CNG",
    "awd": "AWD",
}

# We want to normalize some of the categorical values
# This normalization would involve stripping the un-necessary white-spaced
# characters, lowering the text values to apply casing rules, fixing some
# acronyms, amongst other techniques.
def _normalize_token(tok: str) -> str:
    if tok is None or str(tok).strip() == "":
        return tok
    t_raw = str(tok).strip()

    # Collapse internal multiple spaces
    t = re.sub(r"\s+", " ", t_raw)

    # Lower for matching; then map; then apply casing rules
    low = t.lower()

    # quick spelling correction
    low = low.replace("hyrbrid", "hybrid")

    # direct canonical map if full-token match
    if low in CANON_MAP:
        return CANON_MAP[low]

    # If token is a known two-word phrase but with odd spacing/hyphen
    if re.fullmatch(r"plug[\s-]*in\s*hybrid", low):
        return "Plug-in Hybrid"

    # Generic casing: capitalize words
    # keeps hyphen second part lowercased by design of capitalize()
    t2 = _cap_words_keep_hyphen(t)

    # Title-case content inside parentheses too
    def _cap_inside_paren(m):
        inner = m.group(1)
        return "(" + _cap_words_keep_hyphen(inner) + ")"
    t2 = re.sub(r"\(([^)]*)\)", _cap_inside_paren, t2)

    # Fix acronyms after casing
    t2 = _fix_acronyms(t2)

    return t2

# Normalize the tokens (casing and its spelling)
# But keep the original separators or mix as it is
def normalize_fuel_types_entry(val: str) -> str:
    if pd.isna(val):
        return np.nan
    s = str(val)

    # Collapse overall extra spaces
    # Not around separators yet
    s = re.sub(r"\s+", " ", s.strip())

    # Split into tokens and preserve separators
    # These means preserving the slash or comma with any surrounding spaces
    parts = re.split(r"(\s*[\/,]\s*)", s)

    norm_parts = []
    for p in parts:
        # If this is a separator chunk (slash/comma with spaces), keep as-is
        if re.fullmatch(r"\s*[\/,]\s*", p):
            norm_parts.append(p)
        else:
            norm_parts.append(_normalize_token(p))

    out = "".join(norm_parts)
    out = re.sub(r"\s+", " ", out).strip()
    return out

# We want to add a new column beside the original "Fuel Types" column.
# All future operations will be computed using this new column
def add_fuel_types_normalized_column(df):
    anchor = "Fuel Types"
    df["fuel_types_normalized"] = df[anchor].map(normalize_fuel_types_entry)

    cols = df.columns.tolist()
    if "fuel_types_normalized" in cols:
        cols.remove("fuel_types_normalized")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "fuel_types_normalized")
    return df[cols]

# This helper function is used to split the content of "CC/Battery Capacity" column
# We would be splitting: CC/Battery Capacity to engine_cc & battery_kwh
def _first_number(s):
    if pd.isna(s):
        return np.nan
    m = re.findall(NUM_RE, str(s))
    # take FIRST numeric if multiple appear
    return float(m[0].replace(",", "")) if m else np.nan

def _parse_cc_kwh(val):
    if pd.isna(val):
        return (np.nan, np.nan)
    s = str(val).strip()
    s_low = s.lower()
    has_cc  = bool(re.search(r"\bcc\b", s_low))
    has_kwh = bool(re.search(r"\bkwh\b", s_low))
    if has_cc and not has_kwh:
        return (_first_number(s), np.nan)
    if has_kwh and not has_cc:
        return (np.nan, _first_number(s))
    return (np.nan, np.nan)

# We want to add a new column beside the original "CC/Battery Capacity" column.
# All future operations will be computed using this new column
def add_cc_battery_split_column(df):
    anchor = "CC/Battery Capacity"
    engine_cc_series, battery_kwh_series = zip(*df[anchor].map(_parse_cc_kwh))
    df["engine_displacement_in_cc"] = pd.to_numeric(engine_cc_series, errors="coerce")
    df["battery_energy_capacity_in_kWh"] = pd.to_numeric(battery_kwh_series, errors="coerce")

    # Move the new columns to sit immediately after the anchor column
    cols = list(df.columns)
    anchor_idx = df.columns.get_loc(anchor)
    for new_col in ["engine_displacement_in_cc", "battery_energy_capacity_in_kWh"]:
        if new_col in cols:
            cols.remove(new_col)
    cols[anchor_idx+1:anchor_idx+1] = ["engine_displacement_in_cc", "battery_energy_capacity_in_kWh"]
    return df[cols]

# The "HorsePower" column includes numerical HorsePower values with its associated
# symbol HP or hp. This helper function would help us to strip this column value
#  to remove the text parts, and only retain the numerical value here. We are
# not altering any range components here.
def hp_keep_range(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # strip 'hp'/'HP' tokens
    s = re.sub(r"(?i)\bhp\b", "", s).strip()
    # extract numbers with optional commas/decimals
    nums = re.findall(NUM_RE, s)
    nums = [re.sub(r",", "", n).strip() for n in nums if n.strip() != ""]
    if not nums:
        return np.nan
    # keep range as-is
    if len(nums) >= 2:
        return f"{nums[0]} - {nums[-1]}"
    return nums[0]

# We want to add a new column beside the original "HorsePower" column.
# All future operations will be computed using this new column
def add_horsepower_parsed_column(df):
    anchor = "HorsePower"
    df["horsepower_in_hp"] = df[anchor].map(hp_keep_range)

    cols = df.columns.tolist()
    if "horsepower_in_hp" in cols:
        cols.remove("horsepower_in_hp")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "horsepower_in_hp")
    return df[cols]

# The "Total Speed" column includes numerical speed values with its associated
# symbol in km/h as "180 km/h". This helper function would help us to strip this
# column value to remove the text parts, and only retain the numerical value here.
def speed_to_kmh(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    nums = re.findall(NUM_RE, s)
    if not nums:
        return np.nan
    return float(nums[0].replace(",", ""))

# We want to add a new column beside the original "Total Speed" column.
# All future operations will be computed using this new column
def add_top_speed_column(df):
    anchor = "Total Speed"
    df["total_speed_in_km_per_h"] = df[anchor].map(speed_to_kmh)

    cols = df.columns.tolist()
    if "total_speed_in_km_per_h" in cols:
        cols.remove("total_speed_in_km_per_h")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "total_speed_in_km_per_h")
    return df[cols]

# The "Performance(0 - 100 )KM/H" column includes a time value and a sec text.
# This helper function would help us to strip this column value to remove the
#  text parts, and only retain the numerical value here.
def accel_keep_range(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # strip unit tokens, primarily "sec"; be tolerant to "s"
    s = re.sub(r"(?i)\bsec\b", "", s)
    s = re.sub(r"(?i)\bs\b", "", s).strip()
    nums = re.findall(NUM_RE, s)
    nums = [re.sub(r",", "", n).strip() for n in nums if n.strip() != ""]
    if not nums:
        return np.nan
    if len(nums) >= 2:
        return f"{nums[0]} - {nums[-1]}"
    return nums[0]

# We want to add a new column beside the original "Performance(0 - 100 )KM/H" column.
# All future operations will be computed using this new column
def add_accel_parsed_column(df):
    anchor = "Performance(0 - 100 )KM/H"
    df["performance_0_to_100_km_per_h"] = df[anchor].map(accel_keep_range)

    cols = df.columns.tolist()
    if "performance_0_to_100_km_per_h" in cols:
        cols.remove("performance_0_to_100_km_per_h")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "performance_0_to_100_km_per_h")
    return df[cols]

# The "Torque" column includes the torque value and a Nm text symbol.
# This helper function would help us to strip this column value to remove the
# text parts, and only retain the numerical value here.
def torque_keep_range(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    s = re.sub(r"(?i)\bnm\b", "", s).strip()
    nums = re.findall(NUM_RE, s)
    nums = [re.sub(r",", "", n).strip() for n in nums if n.strip() != ""]
    if not nums:
        return np.nan
    if len(nums) >= 2:
        return f"{nums[0]} - {nums[-1]}"
    return nums[0]

# We want to add a new column beside the original "Torque" column.
# All future operations will be computed using this new column
def add_torque_parsed_column(df):
    anchor = "Torque"
    df["torque_in_nm"] = df[anchor].map(torque_keep_range)

    cols = df.columns.tolist()
    if "torque_in_nm" in cols:
        cols.remove("torque_in_nm")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "torque_in_nm")
    return df[cols]

# These are some essential parsing techniques for the Seats column values
# We want to normalize the dash, keep ranges as it is, and sum the numbers if any.
DASH_CHARS = r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\u2043\x96\x97]"  # hyphen/en/em/minus variants
INT_RE     = re.compile(r"^\s*\d+\s*$")
RANGE_INT  = re.compile(rf"^\s*\d+\s*-\s*\d+\s*$")  # after normalization, hyphen only

def normalize_dashes(s: str) -> str:
    return re.sub(DASH_CHARS, "-", s)

def parse_seats(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # normalize any weird dash to plain '-'
    s = normalize_dashes(s)
    # special case: "2+2" should be added and returned as 4
    if re.fullmatch(r"\s*2\s*\+\s*2\s*", s):
        return 4
    # plain integer to be returned as int value
    if INT_RE.fullmatch(s):
        return int(s)
    # there are some range values like "2-6". We want to keep range as-is.
    if RANGE_INT.fullmatch(s):
        return s
    # anything else should be left as-is (string), but with normalized dashes
    return s

# We want to add a new column beside the original "Seats" column.
# All future operations will be computed using this new column
def add_seats_parsed_column(df):
    anchor = "Seats"
    df["seats_parsed"] = df[anchor].map(parse_seats)

    cols = df.columns.tolist()
    if "seats_parsed" in cols:
        cols.remove("seats_parsed")
    anchor_idx = df.columns.get_loc(anchor)
    cols.insert(anchor_idx + 1, "seats_parsed")
    return df[cols]

# Function to help normalize the column name
# This would also help to normalize the column values in two dataframe columns.
# The columns are: "Company Names" and "Cars Names"
def normalize_colname(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("%", "pct")
    s = s.replace("&", "and")
    s = s.replace("/", "_")
    s = s.replace("’", "").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _normalize_title(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
    return s.title()

def normalize_column_names_and_add_titles(df):
    old_cols = df.columns.tolist()
    new_cols = []
    seen = set()
    for c in old_cols:
        base = normalize_colname(c)
        new = base
        i = 2
        while new in seen:
            new = f"{base}_{i}"
            i += 1
        new_cols.append(new)
        seen.add(new)

    rename_map = dict(zip(old_cols, new_cols))
    df = df.rename(columns=rename_map)

    company_anchor = rename_map.get("Company Names", "company_names")
    cars_anchor    = rename_map.get("Cars Names", "cars_names")

    df["company_name_normalized"] = df[company_anchor].map(_normalize_title)
    df["car_name_normalized"]     = df[cars_anchor].map(_normalize_title)

    cols = df.columns.tolist()
    for anchor, new_col in [(company_anchor, "company_name_normalized"),
                            (cars_anchor,    "car_name_normalized")]:
        if new_col in cols:
            cols.remove(new_col)
        idx = cols.index(anchor)
        cols.insert(idx + 1, new_col)

    df = df[cols]
    return df

def build_polished_dataframe(df):
    desired_cols_raw = [
        'company_name_normalized', 'car_name_normalized', 'engines',
        'engine_displacement_in_cc', 'battery_energy_capacity_in_kwh',
        'horsepower_in_hp', 'total_speed_in_km_per_h',
        'performance_0_to_100_km_per_h', 'cars_price_amount',
        'fuel_types_normalized', 'seats_parsed', 'torque_in_nm'
    ]
    desired_cols = [c for c in desired_cols_raw if c and isinstance(c, str)]
    for c in desired_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[desired_cols].copy()

# These are some of the regular expressions used to match range 
# For a numerical range, we will later be splitting them into multiple rows
# For rows where split is not intended, we will be averaging the numbers
DASH_SPLIT  = re.compile(r"\s*[-–—]\s*")
SLASH_SPLIT = re.compile(r"\s*/\s*")
NUM_RE_TOK  = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def _strip_commas(x: str) -> str:
    return x.replace(",", "") if isinstance(x, str) else x

def _to_float(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    s = s.replace("$", "").replace("€", "")
    s = re.sub(r"(?i)\b(hp|nm|sec|s|kwh|kw|cc|batt|battery)\b", "", s)
    s = _strip_commas(s).strip()
    m = re.findall(NUM_RE_TOK, s)
    if not m:
        return np.nan
    try:
        return float(_strip_commas(m[-1]))
    except Exception:
        return np.nan

def has_dash_range(val) -> bool:
    if pd.isna(val): return False
    s = str(val).strip()
    if not re.search(r"[-–—]", s): return False
    parts = [p for p in re.split(DASH_SPLIT, s) if p.strip()]
    return len(parts) == 2

def dash_range_endpoints(val):
    s = str(val).strip()
    parts = [p.strip() for p in re.split(DASH_SPLIT, s) if p.strip()]
    if len(parts) < 2:
        return (np.nan, np.nan)
    a, b = _to_float(parts[0]), _to_float(parts[-1])
    if pd.isna(a) or pd.isna(b):
        return (np.nan, np.nan)
    return (min(a,b), max(a,b))

def mean_of_dash_range(val):
    a, b = dash_range_endpoints(val)
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return (a + b) / 2.0

def has_slash_variants(val) -> bool:
    if pd.isna(val): return False
    s = str(val).strip()
    return "/" in s and len([p for p in re.split(SLASH_SPLIT, s) if p.strip()]) >= 2

def slash_parts(val):
    s = str(val).strip()
    return [p.strip() for p in re.split(SLASH_SPLIT, s) if p.strip()]

def as_int_if_close(x):
    if pd.isna(x): return np.nan
    return int(round(float(x)))

# helper function for range rule checks across selected columns
def _multi_range_count(row: pd.Series) -> int:
    cols = [
        'horsepower_in_hp', 'cars_price_amount', 'torque_in_nm',
        'performance_0_to_100_km_per_h', 'seats_parsed',
    ]
    return sum(has_dash_range(row[c]) for c in cols if c in row.index)

def row_needs_split(row: pd.Series) -> bool:
    if has_slash_variants(row['engines']):
        return True
    if _multi_range_count(row) >= 2:
        return True
    return False

def row_standalone_averages(row: pd.Series) -> dict:
    updates = {}
    if has_dash_range(row['seats_parsed']):
        updates['seats_parsed'] = as_int_if_close(mean_of_dash_range(row['seats_parsed']))
    if has_dash_range(row['horsepower_in_hp']):
        updates['horsepower_in_hp'] = mean_of_dash_range(row['horsepower_in_hp'])
    if has_dash_range(row['performance_0_to_100_km_per_h']):
        updates['performance_0_to_100_km_per_h'] = mean_of_dash_range(row['performance_0_to_100_km_per_h'])
    if has_dash_range(row['cars_price_amount']):
        updates['cars_price_amount'] = mean_of_dash_range(row['cars_price_amount'])
    if has_dash_range(row['torque_in_nm']):
        updates['torque_in_nm'] = mean_of_dash_range(row['torque_in_nm'])
    return updates

def split_into_two_rows(row: pd.Series) -> list[dict]:
    row1 = row.copy()
    row2 = row.copy()

    if has_slash_variants(row['engines']):
        eng_parts = slash_parts(row['engines'])
        row1['engines'] = eng_parts[0]
        row2['engines'] = eng_parts[-1]

    if has_dash_range(row['seats_parsed']):
        low, high = dash_range_endpoints(row['seats_parsed'])
        row1['seats_parsed'] = as_int_if_close(low)
        row2['seats_parsed'] = as_int_if_close(high)

    val = row['horsepower_in_hp']
    if has_slash_variants(val):
        parts = slash_parts(val)
        row1['horsepower_in_hp'] = _to_float(parts[0])
        row2['horsepower_in_hp'] = _to_float(parts[-1])
    elif has_dash_range(val):
        low, high = dash_range_endpoints(val)
        row1['horsepower_in_hp'] = low
        row2['horsepower_in_hp'] = high

    val = row['performance_0_to_100_km_per_h']
    if has_slash_variants(val):
        parts = slash_parts(val)
        row1['performance_0_to_100_km_per_h'] = _to_float(parts[0])
        row2['performance_0_to_100_km_per_h'] = _to_float(parts[-1])
    elif has_dash_range(val):
        low, high = dash_range_endpoints(val)
        row1['performance_0_to_100_km_per_h'] = low
        row2['performance_0_to_100_km_per_h'] = high

    val = row['cars_price_amount']
    if has_slash_variants(val):
        parts = slash_parts(val)
        row1['cars_price_amount'] = _to_float(parts[0])
        row2['cars_price_amount'] = _to_float(parts[-1])
    elif has_dash_range(val):
        low, high = dash_range_endpoints(val)
        row1['cars_price_amount'] = low
        row2['cars_price_amount'] = high

    val = row['torque_in_nm']
    if has_slash_variants(val):
        parts = slash_parts(val)
        row1['torque_in_nm'] = _to_float(parts[0])
        row2['torque_in_nm'] = _to_float(parts[-1])
    elif has_dash_range(val):
        low, high = dash_range_endpoints(val)
        row1['torque_in_nm'] = low
        row2['torque_in_nm'] = high

    return [row1.to_dict(), row2.to_dict()]

def resolve_ranges_and_splits(df_polished: pd.DataFrame) -> pd.DataFrame:
    base_df = df_polished.copy()
    COLS = [
        'company_name_normalized', 'car_name_normalized', 'engines',
        'engine_displacement_in_cc', 'battery_energy_capacity_in_kwh',
        'horsepower_in_hp', 'total_speed_in_km_per_h',
        'performance_0_to_100_km_per_h', 'cars_price_amount',
        'fuel_types_normalized', 'seats_parsed', 'torque_in_nm'
    ]
    out_rows = []
    for _, row in base_df[COLS].iterrows():
        try:
            if row_needs_split(row):
                out_rows.extend(split_into_two_rows(row))
            else:
                updates = row_standalone_averages(row)
                if updates:
                    row = row.copy()
                    for k, v in updates.items():
                        row[k] = v
                out_rows.append(row.to_dict())
        except Exception:
            out_rows.append(row.to_dict())
    return pd.DataFrame(out_rows)[COLS]

# This is how we handle the missing data
# The missing data will be handled as per the report that we have prepared
def handle_missing_and_fill_sentinels(df_resolved: pd.DataFrame) -> pd.DataFrame:
    df = df_resolved.copy()

    # Drop rows where 0–100 performance is missing
    mask_perf_na = df['performance_0_to_100_km_per_h'].isna()
    df = df.loc[~mask_perf_na].copy()

    # Fill with sentinel -999.0 (keeping indicators)
    SENTINEL = -999.0
    df['battery_energy_capacity_in_kwh'] = df['battery_energy_capacity_in_kwh'].astype('float64').fillna(SENTINEL)
    df['engine_displacement_in_cc']      = df['engine_displacement_in_cc'].astype('float64').fillna(SENTINEL)

    # Leave other missing values (e.g., price/torque singletons) as-is
    return df

def print_missing_table(df: pd.DataFrame, title: str):
    na_cnt = df.isna().sum()
    na_pct = (na_cnt / len(df) * 100).round(1)
    miss_tbl = pd.DataFrame({"missing_count": na_cnt, "missing_pct": na_pct}) \
                 .sort_values("missing_count", ascending=False)
    print(f"\n[{title}]")
    print(miss_tbl.to_string())

def main():
    # we store the data to a dataframe named "cars_dataset"
    cars_dataset = read_csv_robust(
        path_to_dataset,
        keep_default_na=True,
        na_values=extra_na
    )

    # This is intended to do a quick sanity check to know dataset shape and columns.
    print(f"There are a total of {cars_dataset.shape[0]} data values across {cars_dataset.shape[1]} columns")
    print(f"These are all of the column names of the data: {cars_dataset.columns.tolist()}")

    # We want to get more information about the cars_dataset dataframe
    print("\n[General Description of the Dataset]")
    # .describe() prints only numeric by default; include=all gives a fuller picture
    with pd.option_context("display.max_colwidth", None, "display.width", 200):
        print(cars_dataset.describe(include="all").transpose().to_string())

    # We will now randomly view six different rows of data.
    # This is essential to know what operations needs to be performed on what column
    print("\n[Random Sampling 6 rows of the Dataset]")
    print(cars_dataset.sample(6, random_state=42).to_string(index=False))

    # We would want to make a copy of the original dataframe so that the
    # original data is not lost while we work on preprocessing techniques
    df_cars_dataset = cars_dataset.copy()

    # Let's inspect the data patterns of column: "CC/Battery Capacity"
    inspect_patterns(
        df_cars_dataset["CC/Battery Capacity"],
        unit_tokens=["cc","kWh"],
        name="CC/Battery Capacity"
    )

    # Let's inspect the data patterns of column: "HorsePower"
    inspect_patterns(
        df_cars_dataset["HorsePower"],
        unit_tokens=["hp","HP"],
        name="HorsePower"
    )

    # Let's inspect the data patterns of column: "Total Speed"
    inspect_patterns(
        df_cars_dataset["Total Speed"],
        unit_tokens=["km/h"],
        name="Total Speed"
    )

    # Let's inspect the data patterns of column: "Performance(0 - 100 )KM/H"
    inspect_patterns(
        df_cars_dataset["Performance(0 - 100 )KM/H"],
        unit_tokens=["sec","s"],
        name="Performance(0 - 100 )KM/H"
    )

    # Let's inspect the data patterns of column: "Torque"
    inspect_patterns(
        df_cars_dataset["Torque"],
        unit_tokens=["Nm","nm"],
        name="Torque"
    )

    # Let's inspect the data patterns of column: "Total Speed"
    inspect_patterns(
        df_cars_dataset["Total Speed"],
        unit_tokens=["km/h","nm"],
        name="Total Speed"
    )

    # Transformations for each column values
    df_cars_dataset = add_cars_price_amount_column(df_cars_dataset)
    df_cars_dataset = add_fuel_types_normalized_column(df_cars_dataset)
    df_cars_dataset = add_cc_battery_split_column(df_cars_dataset)
    df_cars_dataset = add_horsepower_parsed_column(df_cars_dataset)
    df_cars_dataset = add_top_speed_column(df_cars_dataset)
    df_cars_dataset = add_accel_parsed_column(df_cars_dataset)
    df_cars_dataset = add_torque_parsed_column(df_cars_dataset)
    df_cars_dataset = add_seats_parsed_column(df_cars_dataset)

    # Column name normalization + normalized names
    df_cars_dataset = normalize_column_names_and_add_titles(df_cars_dataset)

    # We will only be keeping the polished columns
    df_cars_dataset_polished = build_polished_dataframe(df_cars_dataset)

    df_resolved = resolve_ranges_and_splits(df_cars_dataset_polished)

    # Details about the missing data before handling the missing values
    print_missing_table(df_resolved, "Missing Data Counter - Start State")

    df_final = handle_missing_and_fill_sentinels(df_resolved)

    # Details about the missing data after handling the missing values
    print_missing_table(df_final, "Missing Data Counter - End State")

    print("\n")
    # We will now randomly view ten different rows of data after normalization
    print(df_final.sample(10, random_state=42).to_string(index=False))

if __name__ == "__main__":
    main()
