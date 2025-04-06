import os.path
import unittest
from pathlib import Path

import pptx2md
import pymupdf4llm

from util.DocumentReader import DocumentReader


class DocumentReaderTest(unittest.TestCase):
    TEST_DIR = f"../../../data/test/{__name__}"

    def test_run_pptx2md(self):
        input_file = "../../../data/earning_calls/msft/MSFT-FY2Q25簡報.pptx"

        if not os.path.exists(self.TEST_DIR):
            os.makedirs(self.TEST_DIR)

        input_path = Path(input_file)
        output_path = Path(self.TEST_DIR) / input_path.with_suffix(".md").name

        config = pptx2md.ConversionConfig(
            pptx_path=input_path,
            output_path=output_path,
            image_dir=None,
            disable_image=True,
        )
        pptx2md.convert(config)

    def test_read_ppt(self):
        input_file = "../../../data/earning_calls/msft/MSFT-FY2Q25簡報.pptx"

        content = DocumentReader.read(input_file)
        print(content)
        self.assertGreater(len(content), 0)  # Read something

    def test_pymupdf4llm(self):
        input_path = Path(self.TEST_DIR) / "test_document.pdf"
        md_text = pymupdf4llm.to_markdown(input_path)

        print(md_text)

        input_path = Path(self.TEST_DIR) / "test_table.pdf"
        md_text = pymupdf4llm.to_markdown(input_path)

        print(md_text)

    def test_read_pdf(self):
        input_file = f"{self.TEST_DIR}/test_document.pdf"
        md_text = DocumentReader.read(input_file)
        print(md_text)

        self.assertGreater(len(md_text), 0)

    def test_read_docx(self):
        input_file = Path(self.TEST_DIR) / "test_word.docx"
        md_text = DocumentReader.read(str(input_file))
        print(md_text)

        self.assertGreater(len(md_text), 0)

    def test_read_xlsx(self):
        input_file = Path(self.TEST_DIR) / "../../專業字詞字典.xlsx"
        md_text = DocumentReader.read(str(input_file))
        print(md_text)

        self.assertGreater(len(md_text), 0)


if __name__ == '__main__':
    unittest.main()
