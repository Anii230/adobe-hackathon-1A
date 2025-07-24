import re

def score_heading_candidate(line, avg_font_size):
    font_info = line["fonts"][0] if line["fonts"] else {}
    font_size = font_info.get("size", 0)
    font_name = font_info.get("font", "")
    font_flags = font_info.get("flags", 0)
    text = line["text"].strip()

    # ---------- Filter noisy/junk lines ----------
    # Skip purely numeric or index-like lines


    # Check for one-word text
    words = text.split()
    if len(words) == 1:
        lower_word = words[0].lower()
        # Skip if it's a known junk field
        if lower_word in {"name", "age", "date", "designation", "relationship", "s.no"}:
            return 0

    if re.match(r"^\d+(\.|:)?$", text):
        return 0

    # Skip common table headers or form fields
    common_labels = {"s.no", "name", "age", "date", "relationship", "designation"}
    if text.lower() in common_labels:
        return 0

    score = 0

    if font_size > avg_font_size:
        score += 1

    if "Bold" in font_name or font_flags in [4, 20]:
        score += 1

    if text.isupper():
        score += 1

    if len(text.split()) <= 10:
        score += 1

    # Bonus: starts with "Section", "Chapter", etc.
    if any(text.lower().startswith(k) for k in ("section", "chapter", "part")):
        score += 1

    return score
