import csv

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
    csv_file_path = "pp_srs.csv"  # Update with the correct path to your CSV file
    integer_part = int(input("Enter the integer part of the timestamp to count (e.g., 5): "))

    try:
        counts, total_count, last_total_count = count_timestamps_by_integer(csv_file_path, integer_part)
        print("Timestamp counts for integer part", integer_part, ":")
        for timestamp, count in counts.items():
            print(f"Timestamp: {timestamp}, Count: {count}")
        print(f"Total count of timestamps with integer part {integer_part}: {total_count}")
        print(f"Total count of all timestamps: {last_total_count}")
    except Exception as e:
        print(f"An error occurred: {e}")