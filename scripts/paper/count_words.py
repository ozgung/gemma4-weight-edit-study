"""Count the words a reader sees in a Markdown writeup: HTML comments removed, everything else counted (title, tables, captions, references), split on whitespace like wc -w.

Usage: python3 -I scripts/paper/count_words.py FILE.md [--sections]
"""
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)  # image links render as images, not words
print(len(text.split()), "words")
if "--sections" in sys.argv:
    for part in re.split(r"\n(?=#{1,3} )", text):
        part = part.strip()
        if part:
            print(f"{len(part.split()):5d}  {part.splitlines()[0][:60]}")
