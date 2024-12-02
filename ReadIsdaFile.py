

import pdfplumber

def process_pdf_file(file_path):
    ignoreStartingPage = 2
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            if ignoreStartingPage > 0:
                ignoreStartingPage -= 1
            else:
                text += page.extract_text()
    return text