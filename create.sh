
# Directories
mkdir -p data/input
mkdir -p data/output
mkdir -p models/MiniLM
mkdir -p src/utils
mkdir -p src/heading_detection
mkdir -p src/section_extraction

# Files
touch README.md
touch requirements.txt
touch Dockerfile
touch approach_explanation.md
touch instructions.txt

touch src/__init__.py

# Utils
touch src/utils/pdf_utils.py
touch src/utils/layout_utils.py
touch src/utils/io_utils.py
touch src/utils/model_utils.py

# Round 1A
touch src/heading_detection/detect_headings.py
touch src/heading_detection/build_outline.py

# Round 1B
touch src/section_extraction/extract_sections.py
touch src/section_extraction/rank_sections.py

# Entrypoint
touch src/main.py
