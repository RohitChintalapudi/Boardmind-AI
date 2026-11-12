"""
backend/app/processing/text_formatter.py
Smart Note Formatting and Structuring Engine for BoardMind AI.

Organizes raw extracted whiteboard text into:
- Main Title
- Headings & Topics
- Bullet Points
- Numbered Steps & Workflows
- Important Definitions
- Formulas, Equations & Key Terms

Strictly preserves original recognized text without inventing facts or hallucinating.
"""

import re
from typing import List, Optional
from backend.app.schemas.models import StructuredNotes


class TextFormatter:
    def format_notes(self, raw_text: str, topic_hint: Optional[str] = None) -> StructuredNotes:
        """
        Structures extracted text into study notes while preserving factual integrity.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        if not lines:
            return StructuredNotes(
                title=topic_hint or "Whiteboard Notes",
                headings=[],
                bullet_points=[],
                numbered_steps=[],
                definitions=[],
                formulas_or_key_terms=[],
                raw_preserved_text="",
                markdown_content="# Whiteboard Notes\n\n*No text was detected on the whiteboard.*"
            )

        headings: List[str] = []
        bullet_points: List[str] = []
        numbered_steps: List[str] = []
        definitions: List[str] = []
        formulas_or_key_terms: List[str] = []

        # Determine title and remaining lines
        is_first_line_math = bool(re.search(r'[\d\w]+\s*[\*\+\-\/\=\^]\s*[\d\w]+', lines[0]))

        if topic_hint:
            title = topic_hint
            remaining_lines = list(lines)
        elif len(lines) == 1:
            if is_first_line_math:
                title = f"Calculation: {lines[0]}"
                remaining_lines = list(lines)
            else:
                title = lines[0].lstrip('#').strip()
                remaining_lines = list(lines)
        else:
            first_line = lines[0]
            if len(first_line) <= 60 and not re.match(r'^(\d+[\.\)]|[-*•])', first_line) and not is_first_line_math:
                title = first_line.lstrip('#').strip()
                remaining_lines = lines[1:]
            else:
                title = "Classroom Lecture Notes"
                remaining_lines = list(lines)

        # Regex patterns
        step_pattern = re.compile(r'^(\d+[\.\)]|\bstep\s*\d+[:\.]?|\([0-9a-zA-Z]\))', re.IGNORECASE)
        bullet_pattern = re.compile(r'^([-*•–—]|->|=>|\+)\s*(.*)')
        def_pattern = re.compile(r'\b(is defined as|means|refers to|definition:?)\b', re.IGNORECASE)
        math_pattern = re.compile(r'(=|\+|-|\*|/|\^|\\approx|\\leq|\\geq|\bformula:?|\bequation:?)\b', re.IGNORECASE)

        for line in remaining_lines:
            # Check for math formula or equation or calculation
            if (
                re.search(r'[\d\w]+\s*[\*\+\-\/\=\^]\s*[\d\w]+', line)
                or ('=' in line and len(line) < 60)
                or math_pattern.search(line)
            ):
                formulas_or_key_terms.append(line.lstrip('-*• ').strip())
                continue

            # Check for definitions
            if def_pattern.search(line) or (':' in line and len(line.split(':', 1)[0].split()) <= 4):
                definitions.append(line.lstrip('-*• ').strip())
                continue

            # Check for numbered steps
            if step_pattern.match(line):
                clean_step = step_pattern.sub('', line).strip()
                numbered_steps.append(clean_step)
                continue

            # Check for bullet points
            bullet_match = bullet_pattern.match(line)
            if bullet_match:
                bullet_points.append(bullet_match.group(2).strip())
                continue

            # Check for headings / topics
            if (len(line) < 45 and not line.endswith('.') and not line.endswith(',')) or line.startswith('#'):
                headings.append(line.lstrip('#').strip())
            else:
                bullet_points.append(line)

        # Build clean markdown
        md_parts = [f"# {title}\n"]

        if definitions:
            md_parts.append("## 📌 Key Definitions")
            for d in definitions:
                if ':' in d:
                    term, desc = d.split(':', 1)
                    md_parts.append(f"- **{term.strip()}:** {desc.strip()}")
                else:
                    md_parts.append(f"- {d}")
            md_parts.append("")

        if formulas_or_key_terms:
            md_parts.append("## 📐 Key Formulas & Calculations")
            md_parts.append("```text")
            for f in formulas_or_key_terms:
                md_parts.append(f)
            md_parts.append("```\n")

        if numbered_steps:
            md_parts.append("## 🔢 Sequential Steps & Procedures")
            for idx, step in enumerate(numbered_steps, 1):
                md_parts.append(f"{idx}. {step}")
            md_parts.append("")

        if bullet_points:
            md_parts.append("## 📝 Core Concepts & Summary")
            for b in bullet_points:
                md_parts.append(f"- {b}")
            md_parts.append("")

        # Attribution footer
        md_parts.append("---\n*Organized automatically by BoardMind AI Note Engine*")

        markdown_content = "\n".join(md_parts)

        return StructuredNotes(
            title=title,
            headings=headings,
            bullet_points=bullet_points,
            numbered_steps=numbered_steps,
            definitions=definitions,
            formulas_or_key_terms=formulas_or_key_terms,
            raw_preserved_text=raw_text,
            markdown_content=markdown_content
        )


# Global singleton formatter instance
_formatter_instance: Optional[TextFormatter] = None

def get_text_formatter() -> TextFormatter:
    global _formatter_instance
    if _formatter_instance is None:
        _formatter_instance = TextFormatter()
    return _formatter_instance
