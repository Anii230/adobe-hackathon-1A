def assign_heading_levels(candidates):
    # Step 1: Get all unique font sizes
    sizes = sorted({line["fonts"][0]["size"] for _, line in candidates if line["fonts"]}, reverse=True)
    size_to_level = {}

    # Step 2: Map sizes to heading levels
    if sizes:
        size_to_level[sizes[0]] = "H1"
    if len(sizes) > 1:
        size_to_level[sizes[1]] = "H2"
    if len(sizes) > 2:
        size_to_level[sizes[2]] = "H3"

    # Step 3: Add level info to each candidate
    outline = []
    for _, line in candidates:
        font_size = line["fonts"][0]["size"]
        level = size_to_level.get(font_size, "Body")
        if level != "Body":
            outline.append({
                "level": level,
                "text": line["text"],
                "page": line["page"]
            })

    return outline
