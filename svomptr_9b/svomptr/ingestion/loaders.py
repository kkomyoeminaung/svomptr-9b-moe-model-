# svomptr/ingestion/loaders.py
import os
import zipfile
import docx
from pypdf import PdfReader
import requests
from bs4 import BeautifulSoup
import io

class FileLoader:
    @staticmethod
    def load_txt(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def load_pdf(path):
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    @staticmethod
    def load_docx(path):
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])

    @staticmethod
    def load_html(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()

    @staticmethod
    def load_zip(path, extract_to):
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return f"Zip file {path} extracted to {extract_to}"
