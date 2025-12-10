
import streamlit as st
from docx import Document
from pypdf import PdfReader
import io

def parse_docx(file):
    """
    Extracts text from a DOCX file.
    
    Args:
        file: A file-like object representing the DOCX file.
        
    Returns:
        str: The extracted text content.
    """
    try:
        document = Document(file)
        full_text = []
        for para in document.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error parsing DOCX file: {e}")
        return None

def parse_pdf(file):
    """
    Extracts text from a PDF file.
    
    Args:
        file: A file-like object representing the PDF file.
        
    Returns:
        str: The extracted text content.
    """
    try:
        reader = PdfReader(file)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text())
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error parsing PDF file: {e}")
        return None

def parse_document(uploaded_file):
    """
    Parses an uploaded document (PDF or DOCX) and extracts its text.
    
    Args:
        uploaded_file: The file object from st.file_uploader.
        
    Returns:
        tuple: (filename, content) or (None, None) if parsing fails.
    """
    if uploaded_file is None:
        return None, None
        
    file_extension = uploaded_file.name.split('.')[-1].lower()
    file_bytes = io.BytesIO(uploaded_file.getvalue())
    
    content = None
    if file_extension == 'pdf':
        content = parse_pdf(file_bytes)
    elif file_extension == 'docx':
        content = parse_docx(file_bytes)
    else:
        st.error("Format file tidak valid. Harap gunakan PDF atau DOCX.")
        return None, None
        
    if content is None or not content.strip():
        st.warning("Could not extract text from the document. The file might be empty, scanned, or corrupted.")
        return uploaded_file.name, ""
        
    return uploaded_file.name, content
