import numpy as np
import argparse

def main(range_end, file_name):
    # Generate numbers
    numbers = np.arange(1, range_end + 1)

    # Shuffle them
    np.random.shuffle(numbers)

    # Write to disk
    np.save(file_name, numbers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and shuffle numbers, then save to a file.")
    
    parser.add_argument("range_end", type=int, help="The end of the range for number generation.")
    parser.add_argument("file_name", type=str, help="The file name where the shuffled numbers will be saved.")

    args = parser.parse_args()

    main(args.range_end, args.file_name)
