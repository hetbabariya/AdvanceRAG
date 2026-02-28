from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


try:
    from langsmith import traceable  # type: ignore
except Exception:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator


class OptimizedPreprocessedLoader:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        preprocessing_config: Optional[Dict[str, bool]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.config: Dict[str, bool] = {
            "remove_extra_whitespace": True,
            "remove_extra_newlines": True,
            "normalize_unicode": True,
            "fix_encoding_errors": True,
            "fix_common_ocr_errors": False,
            "remove_headers_footers": False,
            "remove_page_numbers": False,
            "remove_short_lines": False,
        }

        if preprocessing_config:
            self.config.update(preprocessing_config)

    def load_and_split(self, source: str, custom_metadata: Optional[Dict] = None) -> List[Document]:
        if source.startswith(("http://", "https://")):
            return self._load_from_url(source, custom_metadata or {})
        return self._load_from_file(source, custom_metadata or {})

    def load_and_split_youtube(
        self,
        url: str,
        custom_metadata: Optional[Dict] = None,
        languages: Optional[List[str]] = None,
    ) -> Tuple[str, List[Document]]:
        from urllib.parse import parse_qs, urlparse

        from xml.etree.ElementTree import ParseError

        from youtube_transcript_api import YouTubeTranscriptApi

        custom_metadata = custom_metadata or {}
        languages = languages or ["en", "hi", "en-US", "hi-IN"]

        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]

        video_id = ""
        if host.endswith("youtu.be"):
            video_id = parsed.path.strip("/")
        else:
            qs = parse_qs(parsed.query)
            video_id = (qs.get("v") or [""])[0]

        video_id = (video_id or "").strip()
        if not video_id:
            raise ValueError("Could not extract YouTube video id")

        transcript_items = None
        last_error: Exception | None = None

        for _ in range(2):
            try:
                yt = YouTubeTranscriptApi()
                transcript_items = yt.fetch(video_id, languages=languages)
                break
            except TypeError:
                transcript_items = YouTubeTranscriptApi.get_transcript(video_id)
                break
            except (ParseError, Exception) as e:
                last_error = e

        if not transcript_items:
            if last_error is not None:
                raise ValueError(f"Failed to fetch transcript: {last_error}")
            raise ValueError("No transcript available for this video")

        lines: List[str] = []
        for snippet in transcript_items:
            if isinstance(snippet, dict):
                text = str(snippet.get("text") or "").strip()
            else:
                text = str(getattr(snippet, "text", "") or "").strip()
            if text:
                lines.append(text)

        transcript_text = "\n".join(lines)
        transcript_text = self.preprocess_text(transcript_text)
        if not transcript_text.strip():
            raise ValueError("No transcript available for this video")

        display_name = f"youtube_{video_id}.txt"
        doc_id = self._generate_doc_id(url)

        base_doc = Document(
            page_content=transcript_text,
            metadata={
                "source": url,
                "source_type": "youtube",
                "file_type": "youtube",
                "file_name": display_name,
                "video_id": video_id,
                "loaded_at": datetime.now().isoformat(),
                "doc_id": doc_id,
                **custom_metadata,
            },
        )

        chunks = self.splitter.split_documents([base_doc])
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            chunk.metadata["chunk_id"] = f"{doc_id}_chunk_{i}"
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        return display_name, chunks

    load_and_split_youtube = traceable(name="loader.youtube")(load_and_split_youtube)

    def load_and_split_file(self, file_path: str, original_name: str, custom_metadata: Optional[Dict] = None) -> List[Document]:
        docs = self._load_from_file(file_path, custom_metadata or {})
        for d in docs:
            d.metadata["file_name"] = original_name
        return docs

    def _load_from_url(self, url: str, custom_metadata: Dict) -> List[Document]:
        loader = WebBaseLoader(web_paths=[url])
        docs = loader.load()

        for doc in docs:
            doc.page_content = self.preprocess_text(doc.page_content)
            doc.metadata.update(
                {
                    "source": url,
                    "source_type": "url",
                    "file_type": "html",
                    "loaded_at": datetime.now().isoformat(),
                    "doc_id": self._generate_doc_id(url),
                    **custom_metadata,
                }
            )

        return self._optimized_split(docs)

    def _load_from_file(self, file_path: str, custom_metadata: Dict) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in {".pdf", ".docx"}:
            try:
                return self._load_via_docling(path, custom_metadata)
            except Exception:
                if suffix == ".pdf":
                    page_contents, file_metadata = self._extract_pdf_with_pages(path)
                else:
                    page_contents, file_metadata = self._extract_docx_with_sections(path)
        elif suffix == ".txt":
            page_contents, file_metadata = self._extract_txt(path)
        elif suffix == ".md":
            page_contents, file_metadata = self._extract_md(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        base_metadata = {
            "source": str(path.absolute()),
            "source_type": "file",
            "file_type": suffix[1:],
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "loaded_at": datetime.now().isoformat(),
            "doc_id": self._generate_doc_id(str(path)),
            **file_metadata,
            **custom_metadata,
        }

        docs_with_pages: List[Document] = []
        for page_info in page_contents:
            preprocessed_text = self.preprocess_text(page_info["text"])
            if not preprocessed_text.strip():
                continue

            docs_with_pages.append(
                Document(
                    page_content=preprocessed_text,
                    metadata={
                        **base_metadata,
                        "page_number": page_info["page_num"],
                        "page_start": page_info["page_num"],
                        "page_end": page_info["page_num"],
                    },
                )
            )

        chunks = self._optimized_split_with_pages(docs_with_pages)
        return chunks

    def _load_via_docling(self, path: Path, custom_metadata: Dict) -> List[Document]:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
        pipeline_options.table_structure_options.do_cell_matching = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

        result = converter.convert(str(path))
        doc = result.document
        md = doc.export_to_markdown()
        md = self.preprocess_text(md)
        md = self._normalize_markdown_headings(md)
        if not md.strip():
            return []

        doc_id = self._generate_doc_id(str(path))
        base_metadata = {
            "source": str(path.absolute()),
            "source_type": "file",
            "file_type": path.suffix.lower()[1:],
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "loaded_at": datetime.now().isoformat(),
            "doc_id": doc_id,
            **custom_metadata,
        }

        split_level = int((os.getenv("SECTION_SPLIT_LEVEL", "2") or "2").strip() or "2")
        sections = self._markdown_to_sections(md, split_level=split_level)
        sections = self._merge_small_sections(sections)

        section_docs: List[Document] = []
        for i, section in enumerate(sections):
            title = section.get("title") or ""
            level = int(section.get("level") or 0)
            body = section.get("content") or ""
            content = body
            if title:
                prefix = "#" * max(1, min(level, 6))
                content = f"{prefix} {title}\n\n{body}".strip()

            if not content.strip():
                continue

            section_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        **base_metadata,
                        "section_index": i,
                        "section_title": title,
                        "section_level": level,
                        "section_id": f"{doc_id}_section_{i}",
                    },
                )
            )

        return self._optimized_split(section_docs)

    @staticmethod
    def _markdown_to_sections(md: str, *, split_level: int = 2) -> List[Dict[str, object]]:
        lines = (md or "").splitlines()
        sections: List[Dict[str, object]] = []

        current_title = ""
        current_level = 0
        current_lines: List[str] = []

        heading_re = re.compile(r"^(#{1,6})\s+(.*)\s*$")

        def _flush():
            nonlocal current_title, current_level, current_lines
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    {
                        "title": current_title,
                        "level": current_level,
                        "content": content,
                    }
                )
            current_lines = []

        for line in lines:
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                title = (m.group(2) or "").strip()

                if level <= max(1, min(split_level, 6)):
                    _flush()
                    current_level = level
                    current_title = title
                    continue

                current_lines.append(line)
                continue
            current_lines.append(line)

        _flush()
        if not sections and md.strip():
            return [{"title": "", "level": 0, "content": md.strip()}]
        return sections

    @staticmethod
    def _normalize_markdown_headings(md: str) -> str:
        if not md:
            return ""

        lines = md.splitlines()
        out: List[str] = []
        in_code = False

        heading_inline_re = re.compile(r"#{1,6}")

        for line in lines:
            if line.strip().startswith("```"):
                in_code = not in_code
                out.append(line)
                continue

            if in_code:
                out.append(line)
                continue

            s = line
            parts: List[str] = []
            i = 0
            for m in heading_inline_re.finditer(s):
                idx = m.start()
                if idx == 0:
                    continue

                prev = s[idx - 1]
                if prev.isalnum():
                    continue

                parts.append(s[i:idx])
                parts.append("\n")
                i = idx

            parts.append(s[i:])
            s2 = "".join(parts)

            fixed_lines: List[str] = []
            for ln in s2.splitlines():
                ln2 = re.sub(r"^(#{1,6})(\S)", r"\1 \2", ln)
                fixed_lines.append(ln2)
            out.extend(fixed_lines)

        return "\n".join(out)

    def _merge_small_sections(self, sections: List[Dict[str, object]]) -> List[Dict[str, object]]:
        if not sections:
            return []

        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            def _tok_count(s: str) -> int:
                return len(enc.encode(s or ""))
        except Exception:
            def _tok_count(s: str) -> int:
                return len((s or "").split())

        preamble = sections[0]
        pre_title = str(preamble.get("title") or "")
        pre_level = int(preamble.get("level") or 0)
        pre_content = str(preamble.get("content") or "")

        if pre_title or pre_level != 0:
            return sections

        if len(sections) < 2:
            return sections

        next_sec = dict(sections[1])
        next_content = str(next_sec.get("content") or "")
        joiner = "\n\n" if pre_content.strip() and next_content.strip() else "\n"
        next_sec["content"] = f"{pre_content}{joiner}{next_content}".strip()
        return [next_sec, *sections[2:]]

    def _token_len(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text or ""))
        except Exception:
            return len((text or "").split())

    def _extract_pdf_with_pages(self, path: Path) -> Tuple[List[Dict], Dict]:
        doc = pymupdf.open(path)
        page_contents: List[Dict] = []

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            if page_text.strip():
                page_contents.append({"page_num": page_num, "text": page_text})

        metadata = {
            "page_count": len(doc),
            "author": doc.metadata.get("author", ""),
            "title": doc.metadata.get("title", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
        }

        doc.close()
        metadata = {k: v for k, v in metadata.items() if v}
        return page_contents, metadata

    def _extract_md(self, path: Path) -> Tuple[List[Dict], Dict]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()

        metadata = {
            "character_count": len(text_content),
        }
        return [{"page_num": 1, "text": text_content}], metadata

    def _extract_docx_with_sections(self, path: Path) -> Tuple[List[Dict], Dict]:
        doc = DocxDocument(path)
        words_per_page = 500

        page_contents: List[Dict] = []
        current_page_text: List[str] = []
        current_word_count = 0
        page_num = 1

        for para in doc.paragraphs:
            if not para.text.strip():
                continue
            para_words = len(para.text.split())
            current_page_text.append(para.text)
            current_word_count += para_words

            if current_word_count >= words_per_page:
                page_contents.append({"page_num": page_num, "text": "\n\n".join(current_page_text)})
                current_page_text = []
                current_word_count = 0
                page_num += 1

        if current_page_text:
            page_contents.append({"page_num": page_num, "text": "\n\n".join(current_page_text)})

        for table in doc.tables:
            page_num += 1
            table_text: List[str] = []
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells)
                if row_text.strip():
                    table_text.append(row_text)

            if table_text:
                page_contents.append({"page_num": page_num, "text": "\n".join(table_text)})

        core_props = doc.core_properties
        metadata = {
            "author": core_props.author or "",
            "title": core_props.title or "",
            "subject": core_props.subject or "",
            "keywords": core_props.keywords or "",
            "created": core_props.created.isoformat() if core_props.created else "",
            "modified": core_props.modified.isoformat() if core_props.modified else "",
            "paragraph_count": len(doc.paragraphs),
            "table_count": len(doc.tables),
            "estimated_pages": len(page_contents),
        }

        metadata = {k: v for k, v in metadata.items() if v}
        return page_contents, metadata

    def _extract_txt(self, path: Path) -> Tuple[List[Dict], Dict]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text_content = f.read()

        words_per_page = 500
        words = text_content.split()

        page_contents: List[Dict] = []
        page_num = 1
        current_word_count = 0
        current_page_words: List[str] = []

        for word in words:
            current_page_words.append(word)
            current_word_count += 1

            if current_word_count >= words_per_page:
                page_contents.append({
                    "page_num": page_num,
                    "text": " ".join(current_page_words)
                })
                current_page_words = []
                current_word_count = 0
                page_num += 1

        if current_page_words:
            page_contents.append({
                "page_num": page_num,
                "text": " ".join(current_page_words)
            })

        if not page_contents:
            page_contents.append({"page_num": 1, "text": text_content})

        metadata = {
            "estimated_pages": len(page_contents),
            "word_count": len(words),
            "character_count": len(text_content),
        }

        return page_contents, metadata

    def _extract_html_with_metadata(self, path: Path) -> Tuple[str, Dict]:
        with open(path, "r", encoding="utf-8") as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, "html.parser")

        metadata = {
            "title": soup.title.string if soup.title else "",
            "description": "",
            "keywords": "",
            "author": "",
        }

        for meta in soup.find_all("meta"):
            name = (meta.get("name", "") or "").lower()
            content = meta.get("content", "")

            if name == "description":
                metadata["description"] = content
            elif name == "keywords":
                metadata["keywords"] = content
            elif name == "author":
                metadata["author"] = content

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        metadata = {k: v for k, v in metadata.items() if v}
        return text, metadata

    def _optimized_split(self, docs: List[Document]) -> List[Document]:
        final_chunks: List[Document] = []
        chunk_counter = 0

        for doc in docs:
            section_metadata = dict(doc.metadata)
            if self._token_len(doc.page_content) <= self.chunk_size:
                chunks = [doc.page_content]
            else:
                chunks = self.splitter.split_text(doc.page_content)

            for chunk in chunks:
                final_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **section_metadata,
                            "chunk_index": chunk_counter,
                            "chunk_id": f"{section_metadata.get('doc_id', 'unknown')}_chunk_{chunk_counter}",
                            "chunk_size": len(chunk),
                        },
                    )
                )
                chunk_counter += 1

        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = len(final_chunks)

        return final_chunks

    def _optimized_split_with_pages(self, docs_with_pages: List[Document]) -> List[Document]:
        final_chunks: List[Document] = []
        chunk_counter = 0

        for doc in docs_with_pages:
            page_metadata = dict(doc.metadata)
            if self._token_len(doc.page_content) <= self.chunk_size:
                chunks = [doc.page_content]
            else:
                chunks = self.splitter.split_text(doc.page_content)

            for chunk in chunks:
                final_chunks.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **page_metadata,
                            "chunk_index": chunk_counter,
                            "chunk_id": f"{page_metadata.get('doc_id', 'unknown')}_chunk_{chunk_counter}",
                            "chunk_size": len(chunk),
                        },
                    )
                )
                chunk_counter += 1

        for chunk in final_chunks:
            chunk.metadata["total_chunks"] = len(final_chunks)

        return final_chunks

    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""

        if self.config.get("fix_encoding_errors", False):
            text = self._fix_encoding_errors(text)

        if self.config.get("normalize_unicode", False):
            text = unicodedata.normalize("NFKC", text)

        if self.config.get("fix_common_ocr_errors", False):
            text = self._fix_ocr_errors(text)

        if self.config.get("remove_headers_footers", False):
            text = self._remove_headers_footers(text)

        if self.config.get("remove_page_numbers", False):
            text = self._remove_page_numbers(text)

        if self.config.get("remove_extra_whitespace", False):
            text = re.sub(r"[ \t]+", " ", text)

        if self.config.get("remove_extra_newlines", False):
            text = re.sub(r"\n{3,}", "\n\n", text)

        if self.config.get("remove_short_lines", False):
            lines = text.split("\n")
            lines = [line for line in lines if len(line.split()) >= 3 or line.strip() == ""]
            text = "\n".join(lines)

        return text.strip()

    def _fix_encoding_errors(self, text: str) -> str:
        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€": '"',
            "â€\"": "—",
            "Â": "",
            "\x00": "",
            "\ufffd": "",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _fix_ocr_errors(self, text: str) -> str:
        return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    def _remove_headers_footers(self, text: str) -> str:
        lines = text.split("\n")
        cleaned: List[str] = []

        skip_patterns = [
            r"^page \d+",
            r"^\d+ of \d+$",
            r"^chapter \d+",
            r"^confidential",
            r"^\d+$",
        ]

        for line in lines:
            line_lower = line.lower().strip()
            if any(re.match(p, line_lower) for p in skip_patterns):
                continue
            cleaned.append(line)

        return "\n".join(cleaned)

    def _remove_page_numbers(self, text: str) -> str:
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\bPage\s+\d+\b", "", text, flags=re.IGNORECASE)
        return text

    def _generate_doc_id(self, source: str) -> str:
        return hashlib.md5(source.encode()).hexdigest()[:16]
