#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import List


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def round_cell(cell: str) -> str:
    if is_number(cell):
        return f"{float(cell):.1f}"
    return cell


def build_output_paths(input_path: Path) -> tuple[Path, Path]:
    bev_path = input_path.with_name(f"{input_path.stem}_bev_short.txt")
    d3_path = input_path.with_name(f"{input_path.stem}_3d_short.txt")
    return bev_path, d3_path


def write_aligned_txt(rows: List[List[str]], output_txt: Path) -> None:
    if not rows:
        output_txt.write_text("", encoding="utf-8")
        return

    col_count = max(len(row) for row in rows)
    padded_rows = [row + [""] * (col_count - len(row)) for row in rows]
    col_widths = [max(len(row[i]) for row in padded_rows) for i in range(col_count)]

    lines = []
    for row in padded_rows:
        formatted = "  ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(formatted.rstrip())

    output_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shorten_csv(input_csv: Path, bev_txt: Path, d3_txt: Path) -> None:
    with input_csv.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        rows = [[round_cell(cell) for cell in row] for row in reader]

    if not rows:
        bev_txt.write_text("", encoding="utf-8")
        d3_txt.write_text("", encoding="utf-8")
        return

    header = rows[0]
    data_rows = rows[1:]
    try:
        metric_col = next(i for i, name in enumerate(header) if name.strip().lower() == "metric")
    except StopIteration as exc:
        raise ValueError("CSV is missing a 'metric' column in header.") from exc

    bev_rows = [header]
    d3_rows = [header]
    for row in data_rows:
        metric_value = row[metric_col].strip().lower() if metric_col < len(row) else ""
        if "bev" in metric_value:
            bev_rows.append(row)
        elif "3d" in metric_value:
            d3_rows.append(row)

    write_aligned_txt(bev_rows, bev_txt)
    write_aligned_txt(d3_rows, d3_txt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a CSV into aligned TXT files for BEV and 3D rows with numeric cells rounded to one decimal point."
    )
    parser.add_argument("input_csv", type=Path, help="Path to input CSV file.")
    args = parser.parse_args()

    input_csv = args.input_csv
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    bev_txt, d3_txt = build_output_paths(input_csv)
    shorten_csv(input_csv, bev_txt, d3_txt)
    print(f"Created: {bev_txt}")
    print(f"Created: {d3_txt}")


if __name__ == "__main__":
    main()
