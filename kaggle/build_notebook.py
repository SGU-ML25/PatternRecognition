"""Chuyển skypose_train_kaggle.py (cell '# %%') thành notebook .ipynb."""
from pathlib import Path

import nbformat

src = Path(__file__).with_name("skypose_train_kaggle.py").read_text().splitlines()
cells, kind, buf = [], None, []


def flush():
    text = "\n".join(buf).strip("\n")
    if kind == "md":
        cells.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                                                             for l in text.splitlines())))
    elif kind == "code" and text:
        cells.append(nbformat.v4.new_code_cell(text))


for line in src:
    if line.startswith("# %%"):
        flush()
        kind, buf = ("md" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True},
})
out = Path(__file__).with_name("skypose_train_kaggle.ipynb")
nbformat.write(nb, out)
print(f"{out} ({len(cells)} cells)")
