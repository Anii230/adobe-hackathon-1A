# main.py

import os
import json
import traceback

# Import components from other local files
from pdf_analyser import PDFOutlineExtractor
from utils import INPUT_DIR, OUTPUT_DIR

def main():
    """Main function to run the PDF outline extraction process."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"No PDF files found in the '{INPUT_DIR}' directory.")
            return
    except FileNotFoundError:
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        print("Please create it and place your PDF files inside.")
        return

    for file_name in pdf_files:
        print(f"\nProcessing {file_name}...")
        input_path = INPUT_DIR / file_name
        
        try:
            # Instantiate the extractor and generate the outline
            extractor = PDFOutlineExtractor(input_path)
            result = extractor.generate_outline()
            
            # Save the result to a JSON file
            out_name = os.path.splitext(file_name)[0] + ".json"
            output_path = OUTPUT_DIR / out_name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✔️  Output written to {output_path}")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
