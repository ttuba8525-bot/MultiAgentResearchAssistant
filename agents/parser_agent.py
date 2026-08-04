import fitz  # PyMuPDF


class ParserAgent:
    """
    Extracts text from uploaded PDF files.
    """

    def extract_text(self, pdf_path):
        """
        Extract complete text from a PDF.
        """

        document = fitz.open(pdf_path)

        full_text = ""

        for page in document:
            full_text += page.get_text()

        document.close()

        return full_text

    def extract_sections(self, text):
        """
        Extract important academic sections.
        """

        section_names = [
            "Abstract",
            "Introduction",
            "Related Work",
            "Literature Review",
            "Methodology",
            "Methods",
            "Experimental Setup",
            "Experiments",
            "Results",
            "Discussion",
            "Conclusion",
            "References"
        ]

        sections = {}

        current_section = "Other"

        sections[current_section] = ""

        for line in text.split("\n"):

            cleaned = line.strip()

            if not cleaned:
                continue

            for heading in section_names:

                if cleaned.lower() == heading.lower():

                    current_section = heading

                    if current_section not in sections:
                        sections[current_section] = ""

                    break

            sections[current_section] += cleaned + "\n"

        return sections

    def parse_pdf(self, pdf_path):
        """
        Complete PDF Parsing Pipeline.
        """

        text = self.extract_text(pdf_path)

        sections = self.extract_sections(text)

        return {
            "full_text": text,
            "sections": sections
        }
