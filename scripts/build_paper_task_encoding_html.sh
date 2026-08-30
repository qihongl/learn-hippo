#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
source_file="$repo_root/docs/paper_task_encoding/experiment_report.md"
style_file="$repo_root/docs/learned_encoding/paper.css"
override_style="$repo_root/docs/paper_task_encoding/report_overrides.css"
output_file="$repo_root/docs/paper_task_encoding/experiment_report.html"
title_filter="$repo_root/scripts/pandoc_report_title.lua"

(
  cd "$repo_root/docs/paper_task_encoding"
  pandoc "$(basename "$source_file")" \
    --from=markdown+tex_math_single_backslash \
    --to=html5 \
    --standalone \
    --embed-resources \
    --mathml \
    --toc \
    --toc-depth=2 \
    --lua-filter="$title_filter" \
    --css="$style_file" \
    --css="$override_style" \
    --resource-path="$repo_root/docs/paper_task_encoding:$repo_root" \
    --output="$output_file"
)

python3 - "$output_file" <<'PY'
from html.parser import HTMLParser
from pathlib import Path
import re
import sys


class Validator(HTMLParser):
    def error(self, message):
        raise AssertionError(message)


path = Path(sys.argv[1])
document = path.read_text()
Validator().feed(document)
assert document.count("<img ") == 4, "expected four embedded figures"
assert document.count("<math") >= 6, "expected MathML equations"
assert "data:image/svg+xml" in document, "figures were not embedded"
assert not re.search(r'<(?:img|link|script)[^>]+(?:src|href)="(?!data:)', document), (
    "found a non-embedded runtime resource"
)
assert "measured synthetic" in document.lower()
assert "human review" in document.lower()
assert "the complete success criterion therefore failed" in document.lower()
print(f"Built {path} ({path.stat().st_size:,} bytes)")
PY
