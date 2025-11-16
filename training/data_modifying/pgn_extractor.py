import csv
import argparse
import os


ROW_LIMIT = 45000
def extract_column_to_pgn(csv_path, column_name, output_path, single_file=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if column_name not in reader.fieldnames:
            raise ValueError(f"Column '{column_name}' not found in CSV. Available columns: {reader.fieldnames}")

        count = 0  # Track number of processed rows

        if single_file:
            # Write all rows into one PGN file
            with open(output_path, "w", encoding="utf-8") as pgn_out:
                for row in reader:
                    if count >= ROW_LIMIT:
                        break

                    value = row[column_name].strip()
                    if value:
                        pgn_out.write(value + "\n\n")
                        count += 1

            print(f"Saved combined PGN file to {output_path} ({count} rows written)")

        else:
            # Write each row to a separate PGN file
            base = os.path.splitext(output_path)[0]

            for i, row in enumerate(reader, start=1):
                if count >= ROW_LIMIT:
                    break

                value = row[column_name].strip()
                if not value:
                    continue

                file_name = f"{base}_{i}.pgn"
                with open(file_name, "w", encoding="utf-8") as pgn_out:
                    pgn_out.write(value)

                print(f"Saved {file_name}")
                count += 1


def main():
    parser = argparse.ArgumentParser(description="Extract a CSV column and save as PGN (max 45,000 rows).")
    parser.add_argument("csv_file", help="Input CSV file path")
    parser.add_argument("column", help="Column name to extract")
    parser.add_argument("output", help="Output PGN file path")
    parser.add_argument("--single", action="store_true", help="Save all entries into one PGN file")

    args = parser.parse_args()

    extract_column_to_pgn(args.csv_file, args.column, args.output, args.single)


if __name__ == "__main__":
    main()
