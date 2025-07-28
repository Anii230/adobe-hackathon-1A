# document_profiler.py

import fitz
from collections import Counter
from typing import List, Dict, Tuple

class DocumentProfile:
    """Analyzes and profiles the stylistic elements of a PDF document."""
    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.median_line_spacing = 0.0
        self.lines = self._extract_all_lines()
        self.style_counts = Counter((line['size'], line['is_bold']) for line in self.lines)
        self.body_style = self._find_body_style()

        # Identify potential heading styles based on size and boldness compared to body text
        potential_heading_styles = [
            style for style, count in self.style_counts.items()
            if style[0] > self.body_style[0]
            or (style[0] == self.body_style[0] and style[1] and not self.body_style[1])
        ]
        potential_heading_styles.sort(key=lambda x: (-x[0], -x[1]))

        # Assign heading levels (H1, H2, etc.) based on sorted styles
        self.heading_styles = {}
        if potential_heading_styles:
            base_size = potential_heading_styles[0][0]
            current_level = 1
            for style_idx, (size, is_bold) in enumerate(potential_heading_styles):
                if style_idx == 0:
                    self.heading_styles[(size, is_bold)] = current_level
                else:
                    prev_size, prev_bold = potential_heading_styles[style_idx - 1]
                    # Increment level on significant size drop or change from bold to non-bold
                    if size < prev_size * 0.9 or (not is_bold and prev_bold):
                        current_level += 1
                    self.heading_styles[(size, is_bold)] = current_level

    def _finalize_merge(self, group: List[Dict]) -> Dict:
        """Merges a group of line segments into a single line dictionary."""
        full_text = " ".join(l['text'] for l in group)
        full_bbox = fitz.Rect(group[0]['bbox'])
        for item in group[1:]:
            full_bbox.include_rect(item['bbox'])
        base_line = group[0].copy()
        base_line['text'], base_line['bbox'] = full_text, full_bbox
        return base_line
        
    def _merge_horizontal_lines(self, lines: List[Dict], y_tolerance: float = 2.0, x_tolerance: float = 15.0) -> List[Dict]:
        """Merges text fragments on the same horizontal line into a single entry."""
        if not lines: return []
        lines.sort(key=lambda l: (l['bbox'].y0, l['bbox'].x0))
        merged, current_merge_group = [], [lines[0]]
        for i in range(1, len(lines)):
            prev, curr = current_merge_group[-1], lines[i]
            is_vertically_aligned = abs(curr['bbox'].y0 - prev['bbox'].y0) < y_tolerance
            is_horizontally_close = (curr['bbox'].x0 - prev['bbox'].x1) < x_tolerance and (curr['bbox'].x0 > prev['bbox'].x1)
            
            if is_vertically_aligned and is_horizontally_close:
                current_merge_group.append(curr)
            else:
                merged.append(self._finalize_merge(current_merge_group) if len(current_merge_group) > 1 else current_merge_group[0])
                current_merge_group = [curr]
        
        merged.append(self._finalize_merge(current_merge_group) if len(current_merge_group) > 1 else current_merge_group[0])
        return merged

    def _extract_all_lines(self) -> List[Dict]:
        """Extracts all text lines from the document, merging horizontal fragments."""
        all_lines = []
        for page_num, page in enumerate(self.doc):
            lines_on_page = []
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line_dict in block["lines"]:
                        if not line_dict["spans"]: continue
                        text = " ".join(s["text"] for s in line_dict["spans"]).strip()
                        if not text: continue
                        
                        first_span = line_dict["spans"][0]
                        lines_on_page.append({
                            "text": text,
                            "page": page_num + 1,
                            "size": round(first_span["size"]),
                            "font": first_span["font"],
                            "is_bold": "bold" in first_span["font"].lower() or "black" in first_span["font"].lower(),
                            "bbox": fitz.Rect(line_dict["bbox"]),
                            "line_height": line_dict["bbox"][3] - line_dict["bbox"][1],
                            "space_after": 0.0
                        })
            all_lines.extend(self._merge_horizontal_lines(lines_on_page))
        
        # Calculate vertical spacing between consecutive lines
        spacings = []
        for i in range(len(all_lines) - 1):
            curr_line, next_line = all_lines[i], all_lines[i+1]
            if curr_line['page'] == next_line['page']:
                space = next_line['bbox'].y0 - curr_line['bbox'].y1
                if space > 0:
                    curr_line['space_after'] = space
                    spacings.append(space)
        if spacings:
            self.median_line_spacing = sorted(spacings)[len(spacings) // 2]
        return all_lines

    def _find_body_style(self) -> Tuple[float, bool]:
        """Heuristically determines the style (font size, bold) of the body text."""
        if not self.style_counts: return (12.0, False)
        # Prefer common, non-bold styles in a typical size range
        possible_body_styles = {s: c for s, c in self.style_counts.items() if 9 <= s[0] <= 14 and not s[1]}
        if possible_body_styles:
            return Counter(possible_body_styles).most_common(1)[0][0]
        # Fallback to the most common non-bold style
        non_bold_styles = {s: c for s, c in self.style_counts.items() if not s[1]}
        if non_bold_styles:
            return Counter(non_bold_styles).most_common(1)[0][0]
        # Fallback to the most common style overall
        return self.style_counts.most_common(1)[0][0]
