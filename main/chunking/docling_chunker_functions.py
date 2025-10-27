from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from pathlib import Path
from docling.chunking import HybridChunker

def convert_documents_into_docling_doc(pdf_path: Path):
    pdf_options = PdfPipelineOptions()
    pdf_options.do_ocr = False                    # No OCR - pure text extraction only   # noqa: E501
    pdf_options.generate_page_images = False      # No page images  # noqa: E501
    pdf_options.generate_picture_images = False   # Ignore pictures completely  # noqa: E501
    pdf_options.generate_table_images = False     # Keep tables as text/markdown, not images  # noqa: E501

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
    MAX_TOKENS = 800  # set to a small number for illustrative purposes

    tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
    )
    return tokenizer
    