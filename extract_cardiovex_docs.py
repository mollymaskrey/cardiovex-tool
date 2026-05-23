"""
extract_cardiovex_docs.py

Extracts text from Cardiovex PDFs and saves as .txt files for ingestion.

Usage:
    python extract_cardiovex_docs.py
"""

import pypdf
from pathlib import Path


def extract_pdf_to_txt(pdf_path, output_path):
    """
    Extract all text from a PDF and save to a text file.
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output text file
    """
    print(f"Extracting: {pdf_path.name}")
    
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        num_pages = len(reader.pages)
        print(f"  Pages: {num_pages}")
        
        for page_num, page in enumerate(reader.pages, 1):
            text += page.extract_text()
            text += "\n\n"  # Add spacing between pages
    
    # Write to output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"  Saved: {output_path.name} ({len(text)} characters)")
    return len(text)


def main():
    """Extract all PDFs in data/ directory to docs/ as .txt files."""
    
    # Create directories
    data_dir = Path("data")
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Find all PDFs
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        print("Please place PDF files in the data/ directory")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print("=" * 60)
    
    total_chars = 0
    
    for pdf_path in sorted(pdf_files):
        # Create output filename (remove .pdf, add .txt)
        output_name = pdf_path.stem + ".txt"
        output_path = docs_dir / output_name
        
        chars = extract_pdf_to_txt(pdf_path, output_path)
        total_chars += chars
        print()
    
    print("=" * 60)
    print(f"Extraction complete!")
    print(f"Files extracted: {len(pdf_files)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Output directory: {docs_dir}")


if __name__ == "__main__":
    main()
