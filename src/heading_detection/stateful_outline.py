import fitz  # PyMuPDF
import os
import json
import re
from collections import Counter

INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

# ------------------------ UTILITY FUNCTIONS ------------------------

def extract_lines(pdf_path):
    spans_for_analysis = []
    lines = []
    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line_dict in block["lines"]:
                if not line_dict["spans"]:
                    continue
                first_span = line_dict["spans"][0]
                if first_span["size"] < 7:
                    continue

                line_text = " ".join(s["text"].strip() for s in line_dict["spans"]).strip()
                if not line_text or line_text.lower() in {"version", "table of contents", "page"}:
                    continue

                lines.append({
                    "text": line_text,
                    "page": page_num + 1,
                    "font_size": first_span["size"],
                    "font_name": first_span["font"],
                    "is_bold": any(s["flags"] & 16 for s in line_dict["spans"]),
                    "bbox": line_dict["bbox"],
                    "y0": line_dict["bbox"][1]
                })

                for span in line_dict["spans"]:
                    if span["text"].strip():
                        spans_for_analysis.append({
                            "text": span["text"].strip(),
                            "font_size": span["size"],
                            "is_bold": bool(span["flags"] & 16),
                        })

    doc.close()
    return spans_for_analysis, lines

def get_body_font_size(spans):
    sizes = [
        round(s['font_size']) for s in spans
        if 9 <= s['font_size'] <= 14 and not s['is_bold'] and len(s['text']) > 20
    ]
    return Counter(sizes).most_common(1)[0][0] if sizes else 12.0

def identify_title(lines):
    candidates = [l for l in lines if l['page'] == 1 and l['y0'] < 400]
    if not candidates:
        return "", set()

    max_font = max(l['font_size'] for l in candidates)
    title_lines = [l for l in candidates if l['font_size'] >= 0.9 * max_font and len(l['text'].split()) <= 12]

    title_lines.sort(key=lambda x: x['y0'])
    full_title = " ".join(l['text'].strip() for l in title_lines).strip()
    return full_title, {l['text'] for l in title_lines}

def classify_heading_statefully(line, state, body_font_size):
    text = line['text']
    size = round(line['font_size'], 1)
    is_bold = line['is_bold']
    current_style = (size, is_bold)

    if size <= body_font_size or len(text) < 5:
        return None, state
    if len(text.split()) > 25:
        return None, state
    if re.fullmatch(r"\d+(\.\d+)*", text.strip()):
        return None, state

    numeric_match = re.match(r"^\s*(\d+(\.\d+)*)\.?\s+", text)
    if numeric_match:
        level = numeric_match.group(1).count('.') + 1
        state['level_styles'][level] = current_style
        return f"H{level}", state

    for level, style in state['level_styles'].items():
        if style == current_style:
            return f"H{level}", state

    if not state['level_styles']:
        state['level_styles'][1] = current_style
        return "H1", state

    sorted_levels = sorted(state['level_styles'].items(), key=lambda x: -x[1][0])
    for level, (lvl_size, _) in sorted_levels:
        if size < lvl_size:
            new_level = level + 1
            state['level_styles'][new_level] = current_style
            return f"H{new_level}", state

    state['level_styles'][1] = current_style
    return "H1", state

# ------------------------ OUTLINE GENERATION ------------------------

def generate_outline(pdf_path):
    spans, lines = extract_lines(pdf_path)
    if not lines:
        return {"title": "", "outline": []}

    body_font = get_body_font_size(spans)
    title, title_line_texts = identify_title(lines)
    outline = []
    seen = set()
    state = {"level_styles": {}}

    for line in lines:
        if line['text'] in title_line_texts and line['page'] == 1:
            continue
        if line['bbox'][1] < 50 or line['bbox'][3] > 770:
            continue
        level, state = classify_heading_statefully(line, state, body_font)
        if level:
            clean_text = line['text'].rstrip(' .:').strip()
            key = (clean_text, line['page'])
            if key not in seen:
                outline.append({
                    "level": level,
                    "text": clean_text,
                    "page": line['page']
                })
                seen.add(key)

    return {"title": title, "outline": outline}

# ------------------------ MAIN EXECUTION ------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    for file in pdf_files:
        print(f"Processing {file}...")
        try:
            result = generate_outline(os.path.join(INPUT_DIR, file))
            output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file)[0] + ".json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print(f"✅ Saved to {output_path}")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

if __name__ == "__main__":
    main()
