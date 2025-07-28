# 🧠 Connecting the Dots: Intelligent PDF Outliner (Round 1A)

This repository contains our solution for Round 1A of the **"Connecting the Dots"** challenge. The system is engineered to convert raw PDFs into structured, machine-readable outlines with exceptional speed and accuracy. Given a PDF file, it intelligently identifies the document's **Title** and extracts a hierarchical outline of its headings (H1, H2, H3).

---

## 🚀 Our Approach

Our design avoids brittle, hardcoded rules by implementing a **multi-stage intelligent pipeline** that dynamically adapts to each document's unique structure. The process follows a **"fastest path first"** cascade strategy to maximize both speed and accuracy.

### Table of Contents (ToC) First

The system immediately checks for the most reliable sources of structure:

- **Metadata ToC**: It first prioritizes the PDF's built-in metadata ToC. If present, this provides a perfect, ready-to-use outline.
- **Visual ToC Detection**: If no metadata exists, it scans the initial pages for a visually formatted ToC (e.g., heading text followed by dot leaders and a page number). If detected, it enters a specialized parsing mode for this section.

### Dynamic Document Profiling

If no ToC is found, the system builds a **typographic profile** of the document. By analyzing font sizes, weights, and spacing across all pages, it establishes a data-driven baseline for the document's standard body text. This profile is critical for accurately differentiating headings from paragraphs.

### Flexible Heading Classification

With the document profile as a baseline, every line of text is evaluated by a flexible scoring classifier that combines multiple heuristics:

- **Stylistic**: Is the font larger or bolder than the body text?
- **Layout**: Is there significant vertical whitespace after the line?
- **Content**: Does the line use numbered patterns (e.g., `2.1`), **ALL CAPS**, or Title Case?
- **NLP-Enrichment**: Using `spaCy`, the model analyzes grammatical structure, rewarding noun-heavy phrases common in titles and penalizing complete sentences.

### Hierarchical Assembly

Finally, all identified headings are sorted by page and position. The model enforces a logical hierarchy (e.g., an H3 can only follow an H1 or H2) to produce a clean, structured, and immediately usable **JSON** output.

---

## ✨ Key Features

- **High Accuracy**: The cascading logic and multi-faceted scoring system ensure robust performance across simple and complex PDF layouts.
- **Blazing Fast**: By prioritizing the fastest analysis paths and using the highly optimized PyMuPDF library, the solution easily meets the performance constraints.
- **Multilingual by Design**: Uses `langdetect` to identify the document's language (e.g., English, Japanese) and dynamically loads the appropriate `spaCy` model for nuanced NLP analysis.
- **Fully Offline & Self-Contained**: The Dockerized solution includes all dependencies and models, requiring no network access to run.

---

## 🛠️ Tech Stack

| Component             | Tool/Library                      |
|----------------------|------------------------------------|
| **PDF Parsing**       | `PyMuPDF (fitz)`                   |
| **NLP**               | `spaCy` (`en_core_web_sm`, `ja_core_news_sm`) |
| **Language Detection**| `langdetect`                       |
| **Containerization**  | Docker                             |
| **Language**          | Python 3                           |

---
✅ System Constraints Met
------------------------

| Constraint        | Requirement                      | Status  |
|------------------|-----------------------------------|---------|
| **Execution Time** | ≤ 10 seconds for a 50-page PDF    | ✔️ Met  |
| **Model Size**     | ≤ 200MB                           | ✔️ Met  |
| **Network**        | No internet access                | ✔️ Met  |
| **Runtime**        | CPU-only (amd64)                  | ✔️ Met  |

## 📋 Output JSON Format

The solution outputs a flat JSON structure as specified by the challenge requirements, containing the **title** and a sorted list of outline entries.

```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Introduction", "page": 0 },
    { "level": "H2", "text": "What is AI?", "page": 1 },
    { "level": "H3", "text": "History of AI", "page": 2 }
  ]
}
