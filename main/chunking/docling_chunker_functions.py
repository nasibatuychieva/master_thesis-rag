from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from pathlib import Path
from docling.chunking import HybridChunker

def convert_documents_into_docling_doc(pdf_path: Path):
    ocr_opts = TesseractCliOcrOptions(lang=["eng"])  # OCR nur Englisch

    pdf_options = PdfPipelineOptions(
        do_ocr=True,              # OCR 
        force_full_page_ocr=True, 
        generate_page_images=False,
        generate_table_images=False,
    )

        # Configure format options
    format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_options
            )
        }

    # Initialize document converter
    converter = DocumentConverter(
            format_options=format_options
        )
    result = converter.convert(str(pdf_path))
    doc = result.document
    return doc

def chunk_documents_with_docling(doc, tokenizer):
    chunker = HybridChunker(
    tokenizer=tokenizer,
    merge_peers=True, 
    )
    return chunker

def return_tokenizer():
    EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TOKENS = 800  

    tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  
    )
    return tokenizer
    