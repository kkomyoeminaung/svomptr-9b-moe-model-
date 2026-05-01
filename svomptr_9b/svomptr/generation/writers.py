# svomptr/generation/writers.py
from reportlab.pdfgen import canvas
import docx
import os

class DocWriter:
    @staticmethod
    def write_pdf(path, text):
        c = canvas.Canvas(path)
        c.drawString(100, 750, text)
        c.save()

    @staticmethod
    def write_docx(path, text):
        doc = docx.Document()
        doc.add_paragraph(text)
        doc.save(path)

    @staticmethod
    def write_txt(path, text):
        with open(path, 'w') as f:
            f.write(text)

    @staticmethod
    def write_zip(src_dir, output_path):
        import shutil
        shutil.make_archive(output_path.replace('.zip', ''), 'zip', src_dir)
        return output_path
