# svomptr/generation/writers.py
from reportlab.pdfgen import canvas
import docx
import os

class DocWriter:
    @staticmethod
    def write_pdf(path, text):
        """Writes multi-line PDF using reportlab Paragraph"""
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        style = styles["Normal"]
        
        # Simple line break handling
        p = Paragraph(text.replace('\n', '<br/>'), style)
        doc.build([p])

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
