"""
resume_parser.py — Extract raw text from PDF and DOCX resumes
"""
import io
import re


def parse_pdf(file_obj) -> str:
    """Extract text from a PDF file-like object using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
        if hasattr(file_obj, "read"):
            pdf_bytes = file_obj.read()
        else:
            pdf_bytes = file_obj
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"[PDF Parse Error: {e}]"


def parse_docx(file_obj) -> str:
    """Extract text from a DOCX file-like object using python-docx."""
    try:
        from docx import Document
        doc = Document(file_obj)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(paragraphs).strip()
    except Exception as e:
        return f"[DOCX Parse Error: {e}]"


def parse_resume(file_obj, filename: str) -> str:
    """Auto-detect format and parse resume. Returns extracted text."""
    fname = filename.lower()
    if fname.endswith(".pdf"):
        return parse_pdf(file_obj)
    elif fname.endswith(".docx") or fname.endswith(".doc"):
        return parse_docx(file_obj)
    elif fname.endswith(".txt"):
        return file_obj.read().decode("utf-8", errors="ignore").strip()
    else:
        return "[Unsupported file format. Please upload PDF, DOCX, or TXT.]"


def extract_sections(text: str) -> dict:
    """
    Heuristically split resume text into sections:
    education, experience, skills, projects, summary.
    """
    sections = {
        "summary": "",
        "education": "",
        "experience": "",
        "skills": "",
        "projects": "",
        "other": "",
    }

    section_headers = {
        "summary": ["summary", "objective", "profile", "about me"],
        "education": ["education", "academic", "qualification", "degree"],
        "experience": ["experience", "work experience", "employment", "internship"],
        "skills": ["skill", "technical skill", "core competency", "technology"],
        "projects": ["project", "portfolio", "works"],
    }

    lines = text.split("\n")
    current_section = "other"

    for line in lines:
        stripped = line.strip().lower()
        matched_section = None
        for section, keywords in section_headers.items():
            if any(kw in stripped for kw in keywords):
                if len(stripped) < 60:  # Likely a header
                    matched_section = section
                    break
        if matched_section:
            current_section = matched_section
        else:
            sections[current_section] += line + "\n"

    return {k: v.strip() for k, v in sections.items()}
