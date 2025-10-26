def text2chunk(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks

# print(text2chunk("a b c d e f g h i", 4, 1))