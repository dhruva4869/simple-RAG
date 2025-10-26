from PyPDF2 import PdfReader

class PDFReader:
    def __init__(self, pdf_paths):
        self.pdf_paths = pdf_paths
    
    def read(self, debug=False) -> list[str]:
        texts = []
        for pdf_path in self.pdf_paths:
            with open(pdf_path, "rb") as f:
                pdf = PdfReader(f)
                pages = [page.extract_text() for page in pdf.pages]
                texts.append("\n\n".join(pages))
                if debug:
                    print(pages)
        return texts


def test():
    Reader = PDFReader(["./documents/sample.pdf"])
    texts = Reader.read()
    print(texts[0][:1000])

# test()