import csv
import re
import os
import sys

def log_to_csv(log_file_path, csv_file_path, pattern=None, field_names=None):
    """
    Convert a log file to CSV format
    
    Args:
        log_file_path (str): Path to the log file
        csv_file_path (str): Path to output CSV file
        pattern (str, optional): Regex pattern to extract data. Default is semicolon-separated pattern.
        field_names (list, optional): List of column names for the CSV. Default is predefined field names.
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    # Default pattern for semicolon-separated values
    if pattern is None:
        pattern = r'([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*);([^;]*)'
    
    # Default field names for the CSV
    if field_names is None:
        field_names = [
            'day', 'timestamp', 'product', 
            'bid_price_1', 'bid_volume_1', 
            'bid_price_2', 'bid_volume_2', 
            'bid_price_3', 'bid_volume_3', 
            'ask_price_1', 'ask_volume_1', 
            'ask_price_2', 'ask_volume_2', 
            'ask_price_3', 'ask_volume_3', 
            'mid_price', 'profit_and_loss'
        ]
    
    try:
        # Ensure the log file exists
        if not os.path.isfile(log_file_path):
            print(f"Error: Log file '{log_file_path}' does not exist.")
            return False
        
        # Open the log file and CSV file
        with open(log_file_path, 'r') as log_file, open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            
            # # Write the header
            # csv_writer.writerow(field_names)
            
            # Compile regex pattern
            regex = re.compile(pattern)
            
            line_count = 0
            # Process each line in the log file
            for line in log_file:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                # Extract data using the regex pattern
                match = regex.search(line)
                if match:
                    # Write matching groups to CSV
                    csv_writer.writerow(match.groups())
                else:
                    print(f"Warning: Line {line_count+1} did not match the pattern: {line[:50]}...")
                
                line_count += 1
            
            print(f"Successfully processed {line_count} lines from log file.")
        return True
    
    except Exception as e:
        print(f"Error converting log to CSV: {str(e)}")
        return False

# Main function to handle command-line usage
if __name__ == "__main__":
    
    # Check if correct number of arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python log_to_csv.py <input_log_file> <output_csv_file>")
        sys.exit(1)
    
    # Get input and output file paths from command line arguments
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Call the log_to_csv function
    success = log_to_csv(input_file, output_file)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)