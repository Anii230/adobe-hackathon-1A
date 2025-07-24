from utils.pdf_utils import extract_text_lines
from utils.layout_utils import score_heading_candidate
from heading_detection.build_outline import assign_heading_levels
import statistics

lines = extract_text_lines("data/input/sample.pdf")
font_sizes = [line["fonts"][0]["size"] for line in lines if line["fonts"]]
avg_font_size = statistics.mean(font_sizes)

print("\nTop heading candidates:\n")

# Score and print top candidates
scored = []
for line in lines:
    score = score_heading_candidate(line, avg_font_size)
    if score >= 2:  # You can tune this threshold
        scored.append((score, line))

# Assign heading levels and build outline
outline = assign_heading_levels(scored)

print("\nFinal Outline:")
for item in outline:
    print(f"{item['level']} - Page {item['page']}: {item['text']}")
