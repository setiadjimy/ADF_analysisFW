import re
import csv
from datetime import datetime

input_file = "vehicle_can_data.log"
output_file = "blackbox.log"

# Regex to extract fields
pattern = re.compile(
    r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) '
    r'<0x(?P<canid>[0-9a-fA-F]+)> '
    r'\[(?P<dlc>\d+)\] '
    r'(?P<data>(?:[0-9A-Fa-f]{2} )+)'
    r'.*?lat=(?P<lat>-?\d+\.\d+), lon=(?P<lon>-?\d+\.\d+)'
)

def convert_line(line):
    match = pattern.search(line)
    if not match:
        return None

    timestamp = match.group("timestamp")
    can_id = match.group("canid").upper().zfill(4)  # e.g., 01A0
    dlc = match.group("dlc")

    # Convert timestamp format (comma → dot for milliseconds)
    dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S,%f")
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Convert data bytes to continuous hex string
    data_bytes = match.group("data").strip().replace(" ", "").upper()

    lat = float(match.group("lat"))
    lon = float(match.group("lon"))

    return [
        formatted_time,
        formatted_time,
        f"{lon:.6f}",   # longitude first
        f"{lat:.6f}",   # latitude second
        can_id,
        data_bytes,
        dlc
    ]

# --- MAIN ---
with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)

    for line in infile:
        row = convert_line(line)
        if row:
            writer.writerow(row)

print("Conversion complete!")
