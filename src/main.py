import os
import json
import re
import fitz  # PyMuPDF
import spacy
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- Configuration ---
# Load spaCy model once
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("❌ spaCy model 'en_core_web_sm' not found. Please run:")
    print("   python -m spacy download en_core_web_sm")
    NLP = None

class DocumentProfile:
    """
    Analyzes and stores a comprehensive style and layout profile for a PDF document.
    This enables a highly adaptive classification strategy.
    """
    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.median_line_spacing = 0.0
        self.lines = self._extract_all_lines()

        self.font_counts = Counter(line['size'] for line in self.lines)
        self.style_counts = Counter((line['size'], line['is_bold']) for line in self.lines)

        self.body_size = self._find_body_size()
        self.heading_styles = self._find_heading_styles() # Maps (size, is_bold) -> level

    def _extract_all_lines(self) -> List[Dict]:
        """Extracts all text lines with detailed properties, including vertical spacing."""
        lines = []
        for page_num, page in enumerate(self.doc):
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block["type"] == 0: # Text blocks
                    for line_dict in block["lines"]:
                        if not line_dict["spans"]: continue
                        text = " ".join(s["text"] for s in line_dict["spans"]).strip()
                        if not text: continue
                        
                        first_span = line_dict["spans"][0]
                        line_height = line_dict["bbox"][3] - line_dict["bbox"][1]
                        lines.append({
                            "text": text,
                            "page": page_num + 1,
                            "size": round(first_span["size"]),
                            "font": first_span["font"],
                            "is_bold": "bold" in first_span["font"].lower() or "black" in first_span["font"].lower(),
                            "bbox": fitz.Rect(line_dict["bbox"]),
                            "line_height": line_height,
                            "space_after": 0.0 # Will be calculated next
                        })
        
        # Calculate space_after and median line spacing in a second pass
        spacings = []
        for i in range(len(lines) - 1):
            curr_line = lines[i]
            next_line = lines[i+1]
            if curr_line['page'] == next_line['page']:
                space = next_line['bbox'].y0 - curr_line['bbox'].y1
                if space > 0:
                    curr_line['space_after'] = space
                    spacings.append(space)
        
        if spacings:
            # Use a robust median calculation
            self.median_line_spacing = sorted(spacings)[len(spacings) // 2] if spacings else 0
        
        return lines

    def _find_body_size(self) -> float:
        """Determines the most common font size for non-bold body text."""
        if not self.font_counts: return 12.0
        
        possible_body_sizes = {s: c for (s, is_bold), c in self.style_counts.items() if 9 <= s <= 14 and not is_bold}
        if not possible_body_sizes:
            non_bold_sizes = Counter({s: c for (s, is_bold), c in self.style_counts.items() if not is_bold})
            return non_bold_sizes.most_common(1)[0][0] if non_bold_sizes else self.font_counts.most_common(1)[0][0]

        return Counter(possible_body_sizes).most_common(1)[0][0]

    def _find_heading_styles(self) -> Dict[Tuple[float, bool], int]:
        """Identifies distinct heading levels based on font size and boldness."""
        potential_heading_styles = [style for style in self.style_counts if style[0] > self.body_size]
        potential_heading_styles.sort(key=lambda x: (-x[0], -x[1]))
        return {style: level + 1 for level, style in enumerate(potential_heading_styles)}

class PDFOutlineExtractor:
    """
    Uses an improved layout-aware, multi-pass architecture to achieve high accuracy in outline extraction.
    """
    def __init__(self, pdf_path):
        self.path = pdf_path
        self.doc = fitz.open(pdf_path)
        print("  - Analyzing document styles...")
        self.profile = DocumentProfile(self.doc)
        self.title, self.title_bbox = self._identify_title()
        if self.title:
            print(f"  - Identified Title: '{self.title}'")
        else:
            print("  - No clear title found.")

    def generate_outline(self) -> Dict:
        """Generates the final outline using the best available method."""
        print(f"  - Checking for built-in Table of Contents...")
        toc = self.doc.get_toc()
        if toc and len(toc) > 3:
            print("  - ✔️ Found valid ToC. Using ToC for outline.")
            return self._format_from_toc(toc)
        
        print("  - ⚠️ No valid ToC found. Using feature-based classification...")
        return self._extract_with_classifier()

    def _format_from_toc(self, toc: list) -> Dict:
        """Formats the output from a built-in ToC, ensuring the title is clean."""
        outline = [{
            "level": f"H{level}",
            "text": self._clean_text(text).strip(),
            "page": page
        } for level, text, page in toc]
        return {"title": self.title, "outline": outline}
    
    def _normalize_text_for_comparison(self, text: str) -> str:
        """Creates a consistent key for text comparison by cleaning and normalizing it."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _extract_with_classifier(self) -> Dict:
        """
        Extracts the outline using a feature-based scoring classifier with a centralized de-duplication system.
        """
        headings = []
        # Centralized set to track all text that has been used as a title or heading.
        seen_texts = set()
        
        # Add title to the set first to ensure it's never repeated.
        if self.title:
            seen_texts.add(self._normalize_text_for_comparison(self.title))

        # Determine if header/footer checks should be applied. Disabled for single-page docs.
        check_hf = self.doc.page_count > 1

        # Process all lines using a strict and then a lenient pass if needed.
        for strict_pass in [True, False]:
            if strict_pass and headings: break
            if not strict_pass and headings: break
            if not strict_pass:
                print("  - No headings found in strict mode. Running lenient fallback...")

            for line in self.profile.lines:
                if self.title_bbox and line['page'] == 1 and self.title_bbox.intersects(line['bbox']):
                    continue
                
                classification, level = self._classify_line(line, strict_mode=strict_pass, check_header_footer=check_hf)
                if classification == 'Heading':
                    cleaned_text = self._clean_text(line['text']).strip(' .:-_')
                    text_key = self._normalize_text_for_comparison(cleaned_text)
                    
                    if cleaned_text and text_key not in seen_texts:
                        headings.append({"level": level, "text": cleaned_text, "page": line['page'], "y0": line['bbox'].y0})
                        seen_texts.add(text_key)
        
        outline = self._build_hierarchical_outline(headings)
        return {"title": self.title, "outline": outline}

    def _classify_line(self, line: Dict, strict_mode: bool = True, check_header_footer: bool = True) -> Tuple[str, Optional[int]]:
        """
        Uses a multi-gate system with style and NLP analysis to classify a line.
        """
        text, word_count = line['text'], len(line['text'].split())
        line_style = (line['size'], line['is_bold'])

        # --- Gate 1: Quick Rejection Filters ---
        if check_header_footer:
            page_height = self.doc[line['page'] - 1].rect.height
            if page_height > 0 and (line['bbox'].y0 < page_height * 0.08 or line['bbox'].y1 > page_height * 0.92):
                return "Ignore", None
            
        if word_count > 30: return "Paragraph", None
        if not re.search(r'[a-zA-Z]', self._clean_text(text)): return "Ignore", None
        
        # --- Gate 2: Style Analysis (Mandatory) ---
        style_score = 0
        level = self.profile.heading_styles.get(line_style)
        if level is not None:
            style_score += 10
        elif line['is_bold'] and line['size'] > self.profile.body_size:
            style_score += 4
        
        if style_score < 4:
            return "Paragraph", None

        # --- Gate 3: NLP & Content Analysis (Only in Strict Mode) ---
        nlp_score = 0
        if strict_mode and NLP:
            text_for_nlp = re.sub(r'^\s*(\d{1,2}(\.\d+)*|[A-Z][\.\)]|[a-z][\.\)]|([IVXLCDM]+[\.\)]))\s*', '', text).strip()
            if text_for_nlp and re.search(r'[a-zA-Z]{2,}', text_for_nlp):
                doc = NLP(text_for_nlp)
                if len(list(doc.sents)) > 1 or text_for_nlp.endswith(('.', '!', '?')): nlp_score -= 8
                pos_counts = Counter(token.pos_ for token in doc)
                has_verb = pos_counts['VERB'] > 0 or pos_counts['AUX'] > 0
                is_noun_heavy = (pos_counts['NOUN'] + pos_counts['PROPN']) / len(doc) > 0.5 if len(doc) > 0 else False
                if is_noun_heavy and not has_verb: nlp_score += 8
                elif has_verb: nlp_score -= 5
                elif not is_noun_heavy: nlp_score -= 3
        
        content_score = 0
        if text.isupper() and word_count > 0: content_score += 4 # Increased score
        if self._is_title_case(text) and word_count > 2: content_score += 2
        if text.endswith(('.', ':', ',')): content_score -= 4
        if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s+\d{4}', text, re.IGNORECASE):
            content_score -= 10

        # --- Gate 4: Layout as a final booster ---
        layout_score = 0
        if self.profile.median_line_spacing > 0 and line['space_after'] > self.profile.median_line_spacing * 1.5:
            layout_score += 4

        # --- Final Decision ---
        final_score = style_score + nlp_score + content_score + layout_score
        if strict_mode:
            if final_score >= 11 and nlp_score >= -2:
                return "Heading", level
        else: # Lenient mode only cares about style and layout
            if style_score + layout_score + content_score >= 8:
                return "Heading", level
        
        return "Paragraph", None

    def _identify_title(self) -> Tuple[str, Optional[fitz.Rect]]:
        """
        Identifies the document title by finding the most prominent, concise line(s) at the top of the first page.
        """
        y_limit = self.doc[0].rect.height * 0.40 # Search in the top 40% of the page
        first_page_lines = [line for line in self.profile.lines if line['page'] == 1 and line['bbox'].y1 < y_limit]
        if not first_page_lines: return "", None
        
        max_size = max(line['size'] for line in first_page_lines)
        
        title_candidates = [
            line for line in first_page_lines 
            if line['size'] >= max_size * 0.95 and len(line['text'].split()) < 20
        ]
        if not title_candidates: return "", None

        title_candidates.sort(key=lambda x: x['bbox'].y0)

        title_block_lines = []
        if title_candidates:
            title_block_lines.append(title_candidates[0])
            if len(title_candidates) > 1:
                prev_line = title_candidates[0]
                next_line = title_candidates[1]
                gap = next_line['bbox'].y0 - prev_line['bbox'].y1
                if gap < prev_line['line_height'] * 1.5 and next_line['size'] >= max_size * 0.8:
                    title_block_lines.append(next_line)

        if not title_block_lines: return "", None

        title_bbox = fitz.Rect()
        for line in title_block_lines:
            title_bbox.include_rect(line['bbox'])
        
        final_title = " ".join(self._clean_text(line['text']) for line in title_block_lines)
        
        return self._clean_text(final_title).strip(' .:-_'), title_bbox

    def _is_title_case(self, text: str) -> bool:
        """Checks if a string is likely in Title Case."""
        words = [w for w in text.split() if w.isalpha()]
        if len(words) < 2: return False
        stop_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'of', 'in', 'on', 'to'}
        capitalized = sum(1 for w in words if w[0].isupper() and w.lower() not in stop_words)
        non_stop_word_count = len([w for w in words if w.lower() not in stop_words])
        if non_stop_word_count == 0: return False
        return (capitalized / non_stop_word_count) > 0.6

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing URLs, excessive whitespace, and repeated word patterns."""
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
        return text

    def _build_hierarchical_outline(self, headings: List[Dict]) -> List[Dict]:
        """Sorts and corrects heading levels to build the final outline. De-duplication is handled earlier."""
        if not headings: return []
        
        headings.sort(key=lambda x: (x['page'], x['y0']))
        
        if len(headings) > 1:
            if headings[0].get('level') is None: headings[0]['level'] = 1
            if headings[0]['level'] > 1: headings[0]['level'] = 1
            for i in range(1, len(headings)):
                prev_level = headings[i-1]['level']
                if headings[i].get('level') is None: headings[i]['level'] = prev_level
                if headings[i]['level'] > prev_level + 1:
                    headings[i]['level'] = prev_level + 1

        return [{"level": f"H{h['level']}", "text": h['text'], "page": h['page']} for h in headings]

def main():
    """Main function to process all PDFs in the input directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in the '{INPUT_DIR}' directory.")
        return
    
    for file_name in pdf_files:
        print(f"\nProcessing {file_name}...")
        input_path = INPUT_DIR / file_name
        try:
            extractor = PDFOutlineExtractor(input_path)
            result = extractor.generate_outline()
            
            out_name = os.path.splitext(file_name)[0] + ".json"
            output_path = OUTPUT_DIR / out_name
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"✔️  Output written to {output_path}")
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
