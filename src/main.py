import os
import json
import re
import fitz
import spacy
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")

NLP = None
DETECTED_LANG = None

def load_spacy_model(lang: str):
    global NLP
    if lang == 'en':
        try:
            NLP = spacy.load("en_core_web_sm")
            print("✔️ Loaded spaCy model: 'en_core_web_sm'")
        except OSError:
            print("❌ spaCy model 'en_core_web_sm' not found. Please run:")
            print("    python -m spacy download en_core_web_sm")
            NLP = None
    elif lang == 'ja':
        try:
            NLP = spacy.load("ja_core_news_sm")
            print("✔️ Loaded spaCy model: 'ja_core_news_sm'")
        except OSError:
            print("❌ spaCy model 'ja_core_news_sm' not found. Please run:")
            print("    python -m spacy download ja_core_news_sm")
            NLP = None
    else:
        print(f"⚠️ No spaCy model configured for language: '{lang}'. NLP features will be disabled.")
        NLP = None

class DocumentProfile:
    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.median_line_spacing = 0.0
        self.lines = self._extract_all_lines()
        self.style_counts = Counter((line['size'], line['is_bold']) for line in self.lines)
        self.body_style = self._find_body_style()

        potential_heading_styles = [
            style for style, count in self.style_counts.items()
            if style[0] > self.body_style[0]
            or (style[0] == self.body_style[0] and style[1] and not self.body_style[1])
        ]
        potential_heading_styles.sort(key=lambda x: (-x[0], -x[1]))

        self.heading_styles = {}
        if potential_heading_styles:
            base_size = potential_heading_styles[0][0]
            current_level = 1
            for style_idx, (size, is_bold) in enumerate(potential_heading_styles):
                if style_idx == 0:
                    self.heading_styles[(size, is_bold)] = current_level
                else:
                    prev_size, prev_bold = potential_heading_styles[style_idx - 1]
                    if size < prev_size * 0.9 or (not is_bold and prev_bold):
                        current_level += 1
                    self.heading_styles[(size, is_bold)] = current_level

    def _finalize_merge(self, group: List[Dict]) -> Dict:
        full_text = " ".join(l['text'] for l in group)
        full_bbox = fitz.Rect(group[0]['bbox'])
        for item in group[1:]: full_bbox.include_rect(item['bbox'])
        base_line = group[0].copy()
        base_line['text'], base_line['bbox'] = full_text, full_bbox
        return base_line
        
    def _merge_horizontal_lines(self, lines: List[Dict], y_tolerance: float = 2.0, x_tolerance: float = 15.0) -> List[Dict]:
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
        all_lines = []
        for page_num, page in enumerate(self.doc):
            lines_on_page = []
            blocks = page.get_text("dict", sort=True)["blocks"]
            for block in blocks:
                if block["type"] == 0:
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
        spacings = []
        for i in range(len(all_lines) - 1):
            curr_line, next_line = all_lines[i], all_lines[i+1]
            if curr_line['page'] == next_line['page']:
                space = next_line['bbox'].y0 - curr_line['bbox'].y1
                if space > 0:
                    curr_line['space_after'] = space
                    spacings.append(space)
        if spacings: self.median_line_spacing = sorted(spacings)[len(spacings) // 2]
        return all_lines

    def _find_body_style(self) -> Tuple[float, bool]:
        if not self.style_counts: return (12.0, False)
        possible_body_styles = {s: c for s, c in self.style_counts.items() if 9 <= s[0] <= 14 and not s[1]}
        if possible_body_styles: return Counter(possible_body_styles).most_common(1)[0][0]
        non_bold_styles = {s: c for s, c in self.style_counts.items() if not s[1]}
        if non_bold_styles: return Counter(non_bold_styles).most_common(1)[0][0]
        return self.style_counts.most_common(1)[0][0]

class PDFOutlineExtractor:
    def __init__(self, pdf_path):
        self.path = pdf_path
        self.doc = fitz.open(pdf_path)
        print("  - Analyzing document styles...")
        
        global DETECTED_LANG
        document_sample_text = ""
        for i in range(min(len(self.doc), 3)):
            document_sample_text += self.doc[i].get_text("text")
            if len(document_sample_text) > 500:
                break
        
        if document_sample_text.strip():
            try:
                DETECTED_LANG = detect(document_sample_text)
                print(f"  - Detected document language: {DETECTED_LANG}")
                load_spacy_model(DETECTED_LANG)
            except Exception as e:
                print(f"  - ⚠️ Could not detect language ({e}). Defaulting to English model.")
                DETECTED_LANG = 'en'
                load_spacy_model(DETECTED_LANG)
        else:
            print("  - No discernible text for language detection. Defaulting to English model.")
            DETECTED_LANG = 'en'
            load_spacy_model(DETECTED_LANG)

        self.profile = DocumentProfile(self.doc)
        self.title, self.title_bbox = self._identify_title()
        if self.title: print(f"  - Identified Title: '{self.title}'")
        else: print("  - No clear title found.")

    def generate_outline(self) -> Dict:
        print(f"  - Checking for built-in Table of Contents...")
        toc = self.doc.get_toc()
        if toc:
            print(f"  - ✔️ Found {len(toc)}-item built-in ToC. Using this for outline.")
            return self._format_from_toc(toc)
            
        print(f"  - ⚠️ No built-in ToC. Scanning for dynamic visual ToC...")
        toc_pages = self._find_toc_pages_dynamically()
        if toc_pages:
            print(f"  - ✔️ Identified ToC on page(s): {toc_pages}. Parsing in dedicated ToC mode.")
            full_outline_entries = self._parse_toc_pages(toc_pages)
            if full_outline_entries:
                return {"title": self.title, "outline": self._build_hierarchical_outline(full_outline_entries, is_toc=True)}

        print("  - ⚠️ No parsable ToC found. Using flexible scoring classification...")
        return self._extract_with_classifier()

    def _is_toc_continuation_page(self, page_num: int) -> bool:
        """Checks if a page has the geometry/structure of a ToC page."""
        lines_on_page = [l['text'] for l in self.profile.lines if l['page'] == page_num]
        if len(lines_on_page) < 5: return False
        page_num_re = re.compile(r'(\d+|[ivxlcdm]+)$', re.IGNORECASE)
        match_count = sum(1 for line in lines_on_page if page_num_re.search(line))
        return (match_count / len(lines_on_page)) > 0.6

    def _find_toc_pages_dynamically(self) -> List[int]:
        """Scans for ToC pages and intelligently checks for continuation pages."""
        toc_pages = []
        toc_header_re = re.compile(r'^\s*(table\s+of\s+contents|contents|toc|index|目次|もくじ)\s*$', re.IGNORECASE)
        start_page = -1
        for page_num in range(min(len(self.doc), 6)):
            page_text = self.doc[page_num].get_text("text")
            if toc_header_re.search(page_text):
                start_page = page_num + 1
                break
        if start_page == -1: return []
        toc_pages.append(start_page)
        next_page_num = start_page + 1
        while next_page_num <= len(self.doc):
            if self._is_toc_continuation_page(next_page_num):
                toc_pages.append(next_page_num)
                next_page_num += 1
            else: break
        return toc_pages

    def _parse_toc_pages(self, toc_pages: List[int]) -> List[Dict]:
        """Parses a list of ToC pages using very lenient ToC rules."""
        full_outline_entries = []
        number_prefix_re = re.compile(r'^(\d+(?:\.\d+)*\.?)\s*')
        page_num_re = re.compile(r'(\d+|[ivxlcdm]+)$', re.IGNORECASE)
        toc_header_re = re.compile(r'^\s*(table\s+of\s+contents|contents|toc|index|目次|もくじ)\s*$', re.IGNORECASE)
        for page_num in toc_pages:
            lines_on_page = [l for l in self.profile.lines if l['page'] == page_num]
            for line in lines_on_page:
                line_text = self._clean_text(line['text'])
                if toc_header_re.match(line_text): continue
                page_match = page_num_re.search(line_text)
                if page_match:
                    title_part = line_text[:page_match.start()].rstrip(' .')
                    page_str = page_match.group(1)
                    if len(title_part) > 1:
                        level, number_match = 1, number_prefix_re.match(title_part)
                        if number_match:
                            level = len(re.findall(r'\d+', number_match.group(1)))
                            title_part = title_part[number_match.end():].strip()
                        page_int = self._roman_to_int(page_str) if not page_str.isdigit() else int(page_str)
                        if title_part and page_int > 0:
                            full_outline_entries.append({'level': level, 'text': title_part, 'page': page_int})
        return full_outline_entries

    def _classify_heading_with_score(self, line: Dict) -> Tuple[bool, Optional[int]]:
        """Flexible scoring classifier for non-ToC files."""
        text, word_count = line['text'], len(line['text'].split())
        
        if DETECTED_LANG == 'ja':
            word_count = len(text)
            if word_count > 100 or word_count == 0: return False, None
        else:
            if word_count > 30 or word_count == 0: return False, None

        page_height = self.doc[line['page'] - 1].rect.height
        if page_height > 0 and (line['bbox'].y0 < page_height * 0.05 or line['bbox'].y1 > page_height * 0.95): return False, None
        if not re.search(r'[a-zA-Z\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF\u4E00-\u9FAF]', text):
            return False, None

        score = 0
        style = (line['size'], line['is_bold'])
        body_style = self.profile.body_style
        
        if style[0] > body_style[0] * 1.15: score += 6
        if style[1] and not body_style[1]: score += 4
        
        clean_text = self._clean_text(text)
        
        if DETECTED_LANG == 'en':
            if clean_text.isupper() and word_count > 1: score += 5
            elif self._is_title_case(clean_text): score += 2
            if re.match(r'^\s*(\d+(\.\d+)*\.?|[A-Z]\.)', clean_text): score += 5
        elif DETECTED_LANG == 'ja':
            if re.match(r'^\s*(\d+(\.\d+)*\.?|[一二三四五六七八九十百千万億兆壱弐参肆伍陸漆捌玖拾廿卅|壱弐参四五六七八九〇]+[\.、])', clean_text):
                score += 5
            if word_count < 15: score += 3

        if line['space_after'] > self.profile.median_line_spacing * 1.6: score += 4
        
        if NLP:
            doc = NLP(clean_text)
            if DETECTED_LANG == 'en':
                if len(list(doc.sents)) > 1 or text.endswith(('.', ':', '!', '?')): score -= 6
                content_pos = {'NOUN', 'PROPN', 'ADJ'}
                content_words = sum(1 for t in doc if t.pos_ in content_pos)
                total_words = sum(1 for t in doc if not t.is_punct and not t.is_space)
                if total_words > 0 and (content_words / total_words) < 0.4: score -= 4
            elif DETECTED_LANG == 'ja':
                if text.endswith(('。', '、', '！', '？')): score -= 3
                if len(doc) > 0 and sum(1 for t in doc if not t.is_punct) < 5: score += 2
        
        if score >= 9:
            return True, self.profile.heading_styles.get(style)
        return False, None

    def _extract_with_classifier(self) -> Dict:
        headings, seen_texts = [], set()
        if self.title: seen_texts.add(self._normalize_text_for_comparison(self.title))
        for line in self.profile.lines:
            if self.title_bbox and line['page'] == 1 and line['bbox'].y1 <= self.title_bbox.y1: continue
            is_heading, level = self._classify_heading_with_score(line)
            if is_heading:
                cleaned_text = self._clean_text(line['text'])
                text_key = self._normalize_text_for_comparison(cleaned_text)
                if cleaned_text and text_key not in seen_texts:
                    headings.append({"level": level, "text": cleaned_text, "page": line['page'], "y0": line['bbox'].y0, "size": line['size'], "is_bold": line['is_bold']})
                    seen_texts.add(text_key)
        return {"title": self.title, "outline": self._build_hierarchical_outline(headings)}

    def _format_from_toc(self, toc: list) -> Dict:
        outline = [{"level": f"H{level}", "text": self._clean_text(text).strip(), "page": page} for level, text, page in toc]
        return {"title": self.title, "outline": outline}
        
    def _normalize_text_for_comparison(self, text: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF\u4E00-\u9FAF]', '', text.lower())).strip()

    def _identify_title(self) -> Tuple[str, Optional[fitz.Rect]]:
        page = self.doc[0]
        y_limit = page.rect.height * 0.40
        
        candidate_lines = []
        for line in self.profile.lines:
            if line['page'] == 1 and line['bbox'].y1 < y_limit:
                is_large = line['size'] > self.profile.body_style[0]
                is_bold = line['is_bold'] and not self.profile.body_style[1]
                if is_large or is_bold:
                    candidate_lines.append(line)
        
        if not candidate_lines: return "", None
        
        clusters = []
        if candidate_lines:
            current_cluster = [candidate_lines[0]]
            for i in range(1, len(candidate_lines)):
                prev_line = candidate_lines[i-1]
                curr_line = candidate_lines[i]
                gap = curr_line['bbox'].y0 - prev_line['bbox'].y1
                
                is_close_gap = gap < prev_line['line_height'] * 1.8
                is_same_style = (curr_line['size'] == prev_line['size'] and curr_line['is_bold'] == prev_line['is_bold'])
                
                if is_close_gap and is_same_style:
                    current_cluster.append(curr_line)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [curr_line]
            clusters.append(current_cluster)
        
        best_cluster, max_score = [], -1
        for cluster in clusters:
            if not cluster: continue
            score = 0
            avg_size = sum(l['size'] for l in cluster) / len(cluster)
            score += avg_size * 3
            score -= cluster[0]['bbox'].y0 * 0.1
            
            full_text = " ".join(self._clean_text(l['text']) for l in cluster)
            if NLP and full_text:
                doc = NLP(full_text)
                if DETECTED_LANG == 'en':
                    if len(list(doc.noun_chunks)) > 0 and not any(t.pos_ == 'VERB' for t in doc):
                        score += 15
                elif DETECTED_LANG == 'ja':
                    if any(t.pos_ == 'NOUN' or t.pos_ == 'PROPN' for t in doc) and not any(t.pos_ == 'VERB' for t in doc):
                        score += 15
            
            if score > max_score:
                max_score = score
                best_cluster = cluster
        
        if not best_cluster: return "", None

        final_title = " ".join(self._clean_text(line['text']) for line in best_cluster)
        final_bbox = fitz.Rect(best_cluster[0]['bbox'])
        for line in best_cluster[1:]:
            final_bbox.include_rect(line['bbox'])
            
        return final_title.strip(' .:-_'), final_bbox

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        if DETECTED_LANG == 'en':
            text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
        return text

    def _is_title_case(self, text: str) -> bool:
        if DETECTED_LANG == 'ja': return False
        words = [w for w in text.split() if w.isalpha() and w.lower() not in {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'of', 'in', 'on', 'to'}]
        if not words: return False
        return sum(1 for w in words if w[0].isupper()) / len(words) > 0.6

    def _roman_to_int(self, s: str) -> int:
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        s = s.upper()
        val = 0
        try:
            for i in range(len(s)):
                if i > 0 and roman_map[s[i]] > roman_map[s[i - 1]]: val += roman_map[s[i]] - 2 * roman_map[s[i - 1]]
                else: val += roman_map[s[i]]
        except KeyError: return 0
        return val

    def _build_hierarchical_outline(self, headings: List[Dict], is_toc: bool = False) -> List[Dict]:
        if not headings: return []
        if not is_toc:
            headings.sort(key=lambda x: (x['page'], x['y0']))
            if len(headings) > 1:
                if headings[0].get('level') is None: headings[0]['level'] = 1
                if headings[0].get('level') > 1: headings[0]['level'] = 1
                for i in range(1, len(headings)):
                    prev_level = headings[i-1]['level']
                    if headings[i].get('level') is None: headings[i]['level'] = prev_level
                    if headings[i].get('level') > prev_level + 1:
                        headings[i]['level'] = prev_level + 1
        return [{"level": f"H{h['level']}", "text": self._clean_text(h['text']), "page": h['page']} for h in headings]

def main():
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
