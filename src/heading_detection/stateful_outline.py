import os
import re
import json
import fitz  # PyMuPDF
import spacy
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# Load spaCy NLP model
try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    print("❌ Please install spaCy model with: python -m spacy download en_core_web_sm")
    NLP = None

class DocumentProfile:
    def __init__(self, doc):
        self.doc = doc
        self.font_counts = Counter()
        self.size_to_styles = defaultdict(Counter)
        self.lines = self._extract_lines()
        self._analyze_styles()
        self.body_size = self._find_body_size()
        self.heading_styles = self._find_heading_styles()

    def _extract_lines(self) -> List[Dict]:
        lines = []
        for page_num, page in enumerate(self.doc, start=1):
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text = " ".join(span["text"] for span in spans).strip()
                    if not text:
                        continue
                    first_span = spans[0]
                    lines.append({
                        "text": text,
                        "page": page_num,
                        "size": round(first_span["size"]),
                        "font": first_span["font"],
                        "is_bold": "bold" in first_span["font"].lower(),
                        "bbox": fitz.Rect(line["bbox"])
                    })
        return lines

    def _analyze_styles(self):
        for line in self.lines:
            self.font_counts[line["size"]] += 1
            self.size_to_styles[line["size"]][line["is_bold"]] += 1

    def _find_body_size(self) -> float:
        if not self.font_counts:
            return 12.0
        for size, _ in self.font_counts.most_common():
            if self.size_to_styles[size].get(False, 0) > 0:
                return size
        return self.font_counts.most_common(1)[0][0]

    def _find_heading_styles(self) -> Dict[float, int]:
        sizes = sorted([s for s in self.font_counts if s > self.body_size], reverse=True)
        return {s: i + 1 for i, s in enumerate(sizes)}


class PDFOutlineExtractor:
    def __init__(self, path: Path):
        self.doc = fitz.open(path)
        self.profile = DocumentProfile(self.doc)
        self.title, self.title_bbox = self._identify_title()

    def _identify_title(self) -> Tuple[str, fitz.Rect]:
        title_lines = [l for l in self.profile.lines if l["page"] == 1 and l["size"] >= max(self.profile.font_counts)]
        if not title_lines:
            return "", None
        title_text = " ".join(l["text"] for l in title_lines)
        bbox = title_lines[0]["bbox"]
        for line in title_lines[1:]:
            bbox.include_rect(line["bbox"])
        return title_text, bbox

    def _is_valid_heading(self, line: Dict) -> Tuple[bool, int]:
        level = self.profile.heading_styles.get(line["size"], None)
        if level is None:
            return False, None

        text = line["text"]
        if len(text.split()) > 25 or not re.search(r"[a-zA-Z]", text):
            return False, None

        if NLP:
            doc = NLP(text)
            if len(list(doc.sents)) > 1 or text.endswith("."):
                return False, None
            if not any(tok.pos_ in {"NOUN", "PROPN"} for tok in doc):
                return False, None

        return True, level

    def _clean_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _build_outline(self) -> List[Dict]:
        headings = []
        seen = set()
        for line in self.profile.lines:
            if line["page"] == 1 and self.title_bbox and line["bbox"].intersects(self.title_bbox):
                continue
            is_heading, level = self._is_valid_heading(line)
            if not is_heading:
                continue
            key = (line["text"], line["page"])
            if key in seen:
                continue
            seen.add(key)
            headings.append({
                "text": self._clean_text(line["text"]),
                "page": line["page"],
                "level": f"H{level}"
            })
        return headings

    def extract(self) -> Dict:
        return {
            "title": self._clean_text(self.title),
            "outline": self._build_outline()
        }
