"""Make paste-ready Kaggle Writeup files from a writeup markdown.

    python3 -I scripts/paper/kaggle_writeup.py docs/paper/writeup-3000.md paper1
    python3 -I scripts/paper/kaggle_writeup.py docs/paper2/writeup-resource.md paper2

Writes outputs/paper/kaggle/<name>/{title.txt,subtitle.txt,body.md,figures/}: HTML comments removed, the first
"# " line as the title, the bold line after it as the subtitle, and figure paths pointed at the public study repo
(raw GitHub URLs). The figures are also copied next to the body for manual upload in the Writeup editor.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

RAW = "https://raw.githubusercontent.com/ozgung/gemma4-weight-edit-study/main/figures/"
LOCAL = "../../outputs/paper/figures/"


def main():
    src, name = Path(sys.argv[1]), sys.argv[2]
    out = Path("outputs/paper/kaggle") / name
    (out / "figures").mkdir(parents=True, exist_ok=True)
    text = re.sub(r"<!--.*?-->", "", src.read_text(), flags=re.S)
    lines = text.splitlines()
    i = next(k for k, l in enumerate(lines) if l.startswith("# "))
    title = lines[i][2:].strip()
    j = next(k for k in range(i + 1, len(lines)) if lines[k].strip())
    sub = lines[j].strip()
    assert sub.startswith("**") and sub.endswith("**"), sub
    body = "\n".join(lines[j + 1:])
    for m in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body):
        assert m.startswith(LOCAL), m
        shutil.copy(Path(src.parent, m).resolve(), out / "figures" / Path(m).name)
    body = body.replace(LOCAL, RAW)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"
    assert "<!--" not in body and LOCAL not in body
    (out / "title.txt").write_text(title + "\n")
    (out / "subtitle.txt").write_text(sub.strip("*") + "\n")
    (out / "body.md").write_text(body)
    print(f"{out}: title {len(title)} chars, subtitle {len(sub.strip('*'))} chars, "
          f"{len(re.findall(r'!\[', body))} figures, placeholders {re.findall(r'\[[^\]]*link[^\]]*\]', body)}")


if __name__ == "__main__":
    main()
