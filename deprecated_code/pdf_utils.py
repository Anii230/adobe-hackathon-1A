import fitz  # PyMuPDF

def extract_text_lines(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b["type"] != 0:  # 0 = text block
                continue
            for line in b["lines"]:
                line_text = ""
                fonts = []
                spans = line["spans"]
                for span in spans:
                    line_text += span["text"]
                    fonts.append({
                        "size": span["size"],
                        "font": span["font"],
                        "flags": span["flags"],  # bold/italic info
                        "color": span["color"],
                    })
                if line_text.strip():
                    lines.append({
                        "text": line_text,
                        "fonts": fonts,
                        "page": page_num,
                        "bbox": line["bbox"],  # position
                    })
    return lines
