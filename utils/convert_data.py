import argparse
from data_utils import convert_xlsx_to_jsonl

def main(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    convert_xlsx_to_jsonl(input_path, output_path)
    print("Conversion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XLSX data to JSONL format for fine-tuning.")
    parser.add_argument("--input", type=str, required=True, help="Path to input XLSX file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file.")
    args = parser.parse_args()

    main(args.input, args.output)
