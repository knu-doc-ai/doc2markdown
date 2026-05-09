import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modules.assembly.ir import AssembledDocument, AssemblyMeta, AssemblyResult, ParagraphGroup, SectionNode
from modules.rendering.service import MarkdownRenderer


class MarkdownRenderingTests(unittest.TestCase):
    def test_body_text_starting_with_hash_is_escaped_without_affecting_real_heading(self):
        section = SectionNode(
            id="section_1",
            level=1,
            title="Actual heading",
            heading_block_id="h1",
        )
        paragraph = ParagraphGroup(
            id="paragraph_1",
            block_ids=["p1"],
            text="# literal hash text",
            source_block_ids=["p1"],
        )
        result = AssemblyResult(
            document=AssembledDocument(
                children=[section, paragraph],
                sections=[section],
            ),
            metadata=AssemblyMeta(stage="validated"),
        )

        rendered = MarkdownRenderer().render(result)

        self.assertEqual(rendered.markdown, "# Actual heading\n\n\\# literal hash text")
        self.assertEqual(rendered.warnings, [])

    def test_already_escaped_hash_text_is_not_double_escaped(self):
        paragraph = ParagraphGroup(
            id="paragraph_1",
            block_ids=["p1"],
            text="\\# already escaped",
            source_block_ids=["p1"],
        )
        result = AssemblyResult(
            document=AssembledDocument(children=[paragraph]),
            metadata=AssemblyMeta(stage="validated"),
        )

        rendered = MarkdownRenderer().render(result)

        self.assertEqual(rendered.markdown, "\\# already escaped")

    def test_body_text_starting_with_blockquote_marker_is_escaped(self):
        paragraph = ParagraphGroup(
            id="paragraph_1",
            block_ids=["p1"],
            text="> quoted looking text",
            source_block_ids=["p1"],
        )
        result = AssemblyResult(
            document=AssembledDocument(children=[paragraph]),
            metadata=AssemblyMeta(stage="validated"),
        )

        rendered = MarkdownRenderer().render(result)

        self.assertEqual(rendered.markdown, "\\> quoted looking text")

    def test_thematic_break_like_text_is_escaped(self):
        for raw_text, escaped_text in [
            ("---", "\\---"),
            ("***", "\\***"),
            ("___", "\\___"),
            ("- - -", "\\- - -"),
        ]:
            with self.subTest(raw_text=raw_text):
                paragraph = ParagraphGroup(
                    id="paragraph_1",
                    block_ids=["p1"],
                    text=raw_text,
                    source_block_ids=["p1"],
                )
                result = AssemblyResult(
                    document=AssembledDocument(children=[paragraph]),
                    metadata=AssemblyMeta(stage="validated"),
                )

                rendered = MarkdownRenderer().render(result)

                self.assertEqual(rendered.markdown, escaped_text)


if __name__ == "__main__":
    unittest.main()
