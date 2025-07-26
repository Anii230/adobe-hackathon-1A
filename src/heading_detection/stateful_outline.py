import fitz  # PyMuPDF
import os
import json
import re
from collections import Counter

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
                if not line_dict["spans"] or line_dict["spans"][0]["size"] < 7:
                    continue

                line_text = " ".join(s["text"].strip() for s in line_dict["spans"]).strip()
                if not line_text:
                    continue

                first_span = line_dict["spans"][0]
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
    if not spans:
        return 12.0
    sizes = [
        round(s['font_size']) for s in spans 
        if 9 <= s['font_size'] <= 14 and not s['is_bold'] and len(s['text']) > 20
    ]
    return Counter(sizes).most_common(1)[0][0] if sizes else 12.0

def identify_title(lines):
    candidates = [line for line in lines if line['page'] == 1 and line['y0'] < 400]
    if not candidates:
        return "", set()

    max_font_size = max(c['font_size'] for c in candidates)
    title_lines = [line for line in candidates if line['font_size'] >= max_font_size * 0.9]
    if not title_lines:
        return "", set()
    
    title_lines.sort(key=lambda x: x['y0'])
    full_title = " ".join(line['text'] for line in title_lines if len(line['text'].split()) < 20)
    title_line_texts = {line['text'] for line in title_lines}
    
    return full_title, title_line_texts

def classify_heading_statefully(line, state, body_font_size):
    text = line['text']
    size = round(line['font_size'], 1)
    is_bold = line['is_bold']
    current_style = (size, is_bold)

    if size <= body_font_size or len(text.split()) > 25:
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
    
    sorted_levels = sorted(state['level_styles'].keys())
    for level in sorted_levels:
        level_size, _ = state['level_styles'][level]
        if size < level_size:
            new_level = level + 1
            state['level_styles'][new_level] = current_style
            return f"H{new_level}", state

    state['level_styles'][1] = current_style
    return "H1", state

def generate_outline(pdf_path):
    spans, lines = extract_lines(pdf_path)
    if not lines:
        return {"title": "", "outline": []}

    body_font = get_body_font_size(spans)
    title, title_line_texts = identify_title(lines)

    outline = []
    seen_headings = set()
    document_state = {"level_styles": {}}

    for line in lines:
        if line['bbox'][1] < 50 or line['bbox'][3] > 770:
            continue
        if line['text'] in title_line_texts and line['page'] == 1:
            continue

        level, document_state = classify_heading_statefully(line, document_state, body_font)

        if level:
            cleaned_text = line['text'].rstrip(' .:')
            key = (cleaned_text, line['page'])
            if key not in seen_headings:
                outline.append({
                    "level": level,
                    "text": cleaned_text,
                    "page": line['page']
                })
                seen_headings.add(key)

    return {
        "title": title,
        "outline": outline
    }
