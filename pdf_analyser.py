# pdf_analyser.py

import fitz
import re
from langdetect import detect
from typing import List, Dict, Tuple, Optional

# Import components from other local files
from document_profiler import DocumentProfile
import utils

class PDFOutlineExtractor:
    """
    Extracts a hierarchical outline from a PDF document by analyzing its
    structure, Table of Contents, and text styling.
    """
    def __init__(self, pdf_path):
        self.path = pdf_path
        self.doc = fitz.open(pdf_path)
        print("  - Analyzing document styles...")
        
        # --- Language Detection ---
        document_sample_text = ""
        for i in range(min(len(self.doc), 3)):
            document_sample_text += self.doc[i].get_text("text")
            if len(document_sample_text) > 500: break
        
        if document_sample_text.strip():
            try:
                utils.DETECTED_LANG = detect(document_sample_text)
                print(f"  - Detected document language: {utils.DETECTED_LANG}")
                utils.load_spacy_model(utils.DETECTED_LANG)
            except Exception as e:
                print(f"  - ⚠️ Could not detect language ({e}). Defaulting to English model.")
                utils.DETECTED_LANG = 'en'
                utils.load_spacy_model(utils.DETECTED_LANG)
        else:
            print("  - No discernible text for language detection. Defaulting to English.")
            utils.DETECTED_LANG = 'en'
            utils.load_spacy_model(utils.DETECTED_LANG)

        self.profile = DocumentProfile(self.doc)
        self.title, self.title_bbox = self._identify_title()
        if self.title:
            print(f"  - Identified Title: '{self.title}'")
        else:
            print("  - No clear title found.")

    def generate_outline(self) -> Dict:
        """
        Generates the document outline using a cascading strategy:
        1. Use the built-in PDF Table of Contents (ToC).
        2. Scan for a visually formatted ToC in the first few pages.
        3. Use a flexible scoring classifier on all lines if no ToC is found.
        """
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
        """Checks if a page has the structure of a ToC page (e.g., most lines end with a number)."""
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
        for page_num in range(min(len(self.doc), 6)): # Check first 6 pages
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
            else:
                break
        return toc_pages

    def _parse_toc_pages(self, toc_pages: List[int]) -> List[Dict]:
        """Parses a list of ToC pages using lenient rules to find titles and page numbers."""
        full_outline_entries = []
        number_prefix_re = re.compile(r'^(\d+(?:\.\d+)*\.?)\s*')
        page_num_re = re.compile(r'(\d+|[ivxlcdm]+)$', re.IGNORECASE)
        toc_header_re = re.compile(r'^\s*(table\s+of\s+contents|contents|toc|index|目次|もくじ)\s*$', re.IGNORECASE)

        for page_num in toc_pages:
            lines_on_page = [l for l in self.profile.lines if l['page'] == page_num]
            for line in lines_on_page:
                line_text = utils.clean_text(line['text'])
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
                        
                        page_int = utils.roman_to_int(page_str) if not page_str.isdigit() else int(page_str)
                        if title_part and page_int > 0:
                            full_outline_entries.append({'level': level, 'text': title_part, 'page': page_int})
        return full_outline_entries

    def _classify_heading_with_score(self, line: Dict) -> Tuple[bool, Optional[int]]:
        """Flexible scoring classifier to identify headings in non-ToC files."""
        text, word_count = line['text'], len(line['text'].split())
        
        # Basic disqualifiers
        if utils.DETECTED_LANG == 'ja':
            word_count = len(text)
            if word_count > 100 or word_count == 0: return False, None
        else:
            if word_count > 30 or word_count == 0: return False, None

        page_height = self.doc[line['page'] - 1].rect.height
        if page_height > 0 and (line['bbox'].y0 < page_height * 0.05 or line['bbox'].y1 > page_height * 0.95): return False, None # Header/footer
        if not re.search(r'[a-zA-Z\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF\u4E00-\u9FAF]', text): return False, None # No letters

        score = 0
        style = (line['size'], line['is_bold'])
        body_style = self.profile.body_style
        
        # Style-based scores
        if style[0] > body_style[0] * 1.15: score += 6
        if style[1] and not body_style[1]: score += 4
        
        clean_text = utils.clean_text(text)
        
        # Content-based scores (language-specific)
        if utils.DETECTED_LANG == 'en':
            if clean_text.isupper() and word_count > 1: score += 5
            elif utils.is_title_case(clean_text): score += 2
            if re.match(r'^\s*(\d+(\.\d+)*\.?|[A-Z]\.)', clean_text): score += 5
        elif utils.DETECTED_LANG == 'ja':
            if re.match(r'^\s*(\d+(\.\d+)*\.?|[一二三四五六七八九十百千万億兆壱弐参肆伍陸漆捌玖拾廿卅|壱弐参四五六七八九〇]+[\.、])', clean_text): score += 5
            if word_count < 15: score += 3

        # Layout-based scores
        if line['space_after'] > self.profile.median_line_spacing * 1.6: score += 4
        
        # NLP-based scores (and penalties)
        if utils.NLP:
            doc = utils.NLP(clean_text)
            if utils.DETECTED_LANG == 'en':
                if len(list(doc.sents)) > 1 or text.endswith(('.', ':', '!', '?')): score -= 6
                content_pos = {'NOUN', 'PROPN', 'ADJ'}
                content_words = sum(1 for t in doc if t.pos_ in content_pos)
                total_words = sum(1 for t in doc if not t.is_punct and not t.is_space)
                if total_words > 0 and (content_words / total_words) < 0.4: score -= 4
            elif utils.DETECTED_LANG == 'ja':
                if text.endswith(('。', '、', '！', '？')): score -= 3
                if len(doc) > 0 and sum(1 for t in doc if not t.is_punct) < 5: score += 2
        
        # Final decision
        if score >= 9:
            return True, self.profile.heading_styles.get(style)
        return False, None

    def _extract_with_classifier(self) -> Dict:
        """Applies the scoring classifier to all lines to build an outline."""
        headings, seen_texts = [], set()
        if self.title:
            seen_texts.add(utils.normalize_text_for_comparison(self.title))
        
        for line in self.profile.lines:
            # Skip lines on the first page that are above or part of the title
            if self.title_bbox and line['page'] == 1 and line['bbox'].y1 <= self.title_bbox.y1: continue
            
            is_heading, level = self._classify_heading_with_score(line)
            if is_heading:
                cleaned_text = utils.clean_text(line['text'])
                text_key = utils.normalize_text_for_comparison(cleaned_text)
                if cleaned_text and text_key not in seen_texts:
                    headings.append({"level": level, "text": cleaned_text, "page": line['page'], "y0": line['bbox'].y0, "size": line['size'], "is_bold": line['is_bold']})
                    seen_texts.add(text_key)
        
        return {"title": self.title, "outline": self._build_hierarchical_outline(headings)}

    def _format_from_toc(self, toc: list) -> Dict:
        """Formats the built-in ToC from PyMuPDF into the desired structure."""
        outline = [{"level": f"H{level}", "text": utils.clean_text(text).strip(), "page": page} for level, text, page in toc]
        return {"title": self.title, "outline": outline}

    def _identify_title(self) -> Tuple[str, Optional[fitz.Rect]]:
        """Identifies the document title from the first page."""
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
        
        # Cluster adjacent lines with the same style
        clusters = []
        if candidate_lines:
            current_cluster = [candidate_lines[0]]
            for i in range(1, len(candidate_lines)):
                prev_line, curr_line = candidate_lines[i-1], candidate_lines[i]
                gap = curr_line['bbox'].y0 - prev_line['bbox'].y1
                is_close_gap = gap < prev_line['line_height'] * 1.8
                is_same_style = (curr_line['size'] == prev_line['size'] and curr_line['is_bold'] == prev_line['is_bold'])
                
                if is_close_gap and is_same_style:
                    current_cluster.append(curr_line)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [curr_line]
            clusters.append(current_cluster)
        
        # Score clusters to find the best title candidate
        best_cluster, max_score = [], -1
        for cluster in clusters:
            if not cluster: continue
            score = 0
            avg_size = sum(l['size'] for l in cluster) / len(cluster)
            score += avg_size * 3  # Larger text is better
            score -= cluster[0]['bbox'].y0 * 0.1 # Higher on page is better
            
            full_text = " ".join(utils.clean_text(l['text']) for l in cluster)
            if utils.NLP and full_text:
                doc = utils.NLP(full_text)
                # Boost score if the phrase is noun-heavy and lacks verbs
                if utils.DETECTED_LANG == 'en':
                    if len(list(doc.noun_chunks)) > 0 and not any(t.pos_ == 'VERB' for t in doc):
                        score += 15
                elif utils.DETECTED_LANG == 'ja':
                    if any(t.pos_ == 'NOUN' or t.pos_ == 'PROPN' for t in doc) and not any(t.pos_ == 'VERB' for t in doc):
                        score += 15
            
            if score > max_score:
                max_score, best_cluster = score, cluster
        
        if not best_cluster: return "", None

        final_title = " ".join(utils.clean_text(line['text']) for line in best_cluster)
        final_bbox = fitz.Rect(best_cluster[0]['bbox'])
        for line in best_cluster[1:]:
            final_bbox.include_rect(line['bbox'])
            
        return final_title.strip(' .:-_'), final_bbox

    def _build_hierarchical_outline(self, headings: List[Dict], is_toc: bool = False) -> List[Dict]:
        """Builds a structured outline, inferring/correcting heading levels."""
        if not headings: return []
        
        if not is_toc:
            headings.sort(key=lambda x: (x['page'], x['y0']))
            # Post-process to enforce hierarchy rules
            if len(headings) > 1:
                if headings[0].get('level') is None: headings[0]['level'] = 1
                if headings[0].get('level') > 1: headings[0]['level'] = 1 # First heading must be level 1
                for i in range(1, len(headings)):
                    prev_level = headings[i-1]['level']
                    if headings[i].get('level') is None: headings[i]['level'] = prev_level
                    # Prevent jumping levels (e.g., from H1 to H3)
                    if headings[i].get('level') > prev_level + 1:
                        headings[i]['level'] = prev_level + 1
        
        return [{"level": f"H{h['level']}", "text": utils.clean_text(h['text']), "page": h['page']} for h in headings]
