import os
from functools import partial
from pathlib import Path
from typing import Callable, List

import pandas as pd
import pptx2md
import pymupdf4llm
from html2text import HTML2Text
from unstructured.partition.auto import partition


class DocumentReader:

    @classmethod
    def read_text(cls, path: str, **kwargs) -> str:
        supported_extensions = kwargs.pop("supported_extensions", [])
        if all([not path.endswith(ext) for ext in supported_extensions]):
            raise ValueError(f"{path} is not one of {', '.join(supported_extensions)}")

        with open(path, "r") as fd:
            return fd.read()

    @classmethod
    def read_pdf(cls, path: str, **kwargs) -> str:
        supported_extensions = kwargs.pop("supported_extensions", [])
        if all([not path.endswith(ext) for ext in supported_extensions]):
            raise ValueError(f"{path} is not one of {', '.join(supported_extensions)}")

        input_path = Path(path)
        return pymupdf4llm.to_markdown(input_path)

    @classmethod
    def read_ppt(cls, path: str, **kwargs) -> str:

        supported_extensions = kwargs.pop("supported_extensions", [])
        if all([not path.endswith(ext) for ext in supported_extensions]):
            raise ValueError(f"{path} is not one of {', '.join(supported_extensions)}")

        input_path = Path(path)
        output_dir = kwargs.pop("text_directory", "/tmp")
        output_path = Path(output_dir) / input_path.with_suffix(".md").name

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config = pptx2md.ConversionConfig(
            pptx_path=input_path,
            output_path=output_path,
            image_dir=None,
            disable_image=True,
        )
        pptx2md.convert(config)

        md_content = output_path.read_text(encoding="utf-8")
        os.remove(output_path)
        return md_content

    @classmethod
    def read_docx(cls, path: str, **kwargs) -> str:
        supported_extensions = kwargs.pop("supported_extensions", [])
        if all([not path.endswith(ext) for ext in supported_extensions]):
            raise ValueError(f"{path} is not one of {', '.join(supported_extensions)}")

        h2t = HTML2Text()

        markdown_parts = []
        elements = partition(filename=path)
        for element in elements:
            if element.category == "Title":
                depth = "#" * (element.metadata.category_depth + 1)
                markdown_parts.append(f"{depth} {element.text}\n")

            elif element.category == "ListItem":
                depth = "    " * element.metadata.category_depth
                markdown_parts.append(f"{depth}- {element.text}\n")

            elif element.category == "Table":
                table_html = getattr(element.metadata, 'text_as_html', None)
                if table_html:
                    table_md = h2t.handle(table_html)
                    table_md = '\n'.join([f"| {row}" for row in table_md.split("\n") if row])
                    markdown_parts.append(f"\n{table_md}\n")
                else:
                    markdown_parts.append(f"\n[Table: {element.text[:50]}...]\n")
            else:   # Default to paragraph text (NarrativeText etc.)
                markdown_parts.append(f"{element.text}\n")

        return "\n".join(markdown_parts)

    @classmethod
    def df2md(cls, df: pd.DataFrame) -> str:
        """Convert a Pandas dataframe into markdown table."""
        drop_condition = isinstance(df.index, pd.RangeIndex) and df.index.name is None
        df = df.reset_index(drop=drop_condition)
        return df.to_markdown(index=False)

    @classmethod
    def read_xlsx(cls, path: str, **kwargs) -> str:
        supported_extensions = kwargs.pop("supported_extensions", [])
        if all([not path.endswith(ext) for ext in supported_extensions]):
            raise ValueError(f"{path} is not one of {', '.join(supported_extensions)}")

        df = pd.read_excel(path, **kwargs)

        if isinstance(df, pd.DataFrame):
            return cls.df2md(df)

        md_text = ""
        for sheet_name, table in df:  # Multiple sheets
            md_text += f"\n{sheet_name}:\n"
            md_text += cls.df2md(table)

        return md_text

    @classmethod
    def read(cls, path: str, **kwargs) -> str:
        readers: List[Callable[[str], str]] = [
            partial(cls.read_text, supported_extensions=[".txt", ".md"], **kwargs),
            partial(cls.read_pdf, supported_extensions=[".pdf"], **kwargs),
            partial(cls.read_ppt, supported_extensions=[".ppt", ".pptx"], **kwargs),
            partial(cls.read_docx, supported_extensions=[".docx"], **kwargs),
            partial(cls.read_xlsx, supported_extensions=[".xls", ".xlsx"], **kwargs),
        ]

        for r in readers:
            try:
                return r(path)
            except ValueError:
                pass

        raise ValueError(f"Unsupported file type {path}")

