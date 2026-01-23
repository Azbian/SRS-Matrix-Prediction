import csv
import argparse

def count_timestamps_by_integer(csv_file_path, integer_part):
    timestamp_counts = {}
    total_count = 0
    last_total_count = 0

    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            timestamp = float(row['timestamp'])
            integer = int(timestamp)  # Extract the integer part

            if integer == integer_part:
                total_count += 1
                if timestamp in timestamp_counts:
                    timestamp_counts[timestamp] += 1
                else:
                    timestamp_counts[timestamp] = 1

            last_total_count += 1

    return timestamp_counts, total_count, last_total_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count timestamps in a CSV file by integer part.")
    parser.add_argument("--csv_file_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--integer_part", type=int, required=True, help="The integer part of the timestamp to count.")

    args = parser.parse_args()

    try:
        counts, total_count, last_total_count = count_timestamps_by_integer(args.csv_file_path, args.integer_part)
        print("Timestamp counts for integer part", args.integer_part, ":")
        for timestamp, count in counts.items():
            print(f"Timestamp: {timestamp}, Count: {count}")
        print(f"Total count of timestamps with integer part {args.integer_part}: {total_count}")
        print(f"Total count of all timestamps: {last_total_count}")
    except Exception as e:
        print(f"An error occurred: {e}")