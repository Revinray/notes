import pdf2image
import pytesseract
from PyPDF2 import PdfReader
import pandas as pd
import os

# Set paths for Tesseract
pytesseract.pytesseract.tesseract_cmd = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin', 'tesseract.exe')

# Function to convert a PDF page to an image
def pdf_page_to_image(pdf_path, page_num):
    images = pdf2image.convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
    return images[0]

# Function to convert all PDF pages to images
def pages_to_images(pdf_path, num_pages):
    images = []
    for page_num in range(num_pages):
        images.append(pdf_page_to_image(pdf_path, page_num))
    return images

# Function to extract text from PDF and store in DataFrame
def extract_text_into_table(pdf_file_path):
    pdf_file_name = os.path.basename(pdf_file_path)
    pdf_path = "temp_pdf.pdf"
    with open(pdf_file_path, "rb") as pdf_file:
        pdf_content = pdf_file.read()
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    pdf_reader = PdfReader(pdf_path)
    page_obj = pdf_reader.pages[0]
    text = page_obj.extract_text()
    count_of_lines = len(text.split("\n"))
    is_searchable = count_of_lines > 15

    if is_searchable:
        searchable_file = {}
        for page_no in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_no]
            text = page.extract_text()
            searchable_file[str(page_no)] = {"content": text, "pdf_file_name": pdf_file_name}
        df = pd.DataFrame.from_dict(searchable_file, orient="index")
        df.reset_index(inplace=True)
        df.columns = ["page_no", "content", "pdf_file_name"]
        df = df[["page_no", "content", "pdf_file_name"]]
        return df
    else:
        images = pages_to_images(pdf_path, len(pdf_reader.pages))
        scanned_file = {}
        for i in range(len(images)):
            scanned_file[str(i)] = {"content": pytesseract.image_to_string(images[i]), "pdf_file_name": pdf_file_name}
        df = pd.DataFrame.from_dict(scanned_file, orient="index")
        df.reset_index(inplace=True)
        df.columns = ["page_no", "content", "pdf_file_name"]
        df = df[["page_no", "content", "pdf_file_name"]]
        return df
    
