import os, glob

files = glob.glob("analysis/*.py") + glob.glob("tests/*.py")
for f in files:
    s = open(f, encoding="utf-8").read()
    s = s.replace(".write_text(", ".write_text(encoding='utf-8', data=")
    s = s.replace(".read_text()", ".read_text(encoding='utf-8')")
    # fix double-apply
    s = s.replace("encoding='utf-8', data=encoding='utf-8', data=", "encoding='utf-8', data=")
    s = s.replace("encoding='utf-8', encoding='utf-8',", "encoding='utf-8',")
    open(f, "w", encoding="utf-8").write(s)
    print("fixed", f)

# fix NaN in ROC curve for eval_controls
f = "analysis/eval_controls.py"
s = open(f, encoding="utf-8").read()
s = s.replace(
    "fpr, tpr, _ = roc_curve(pos, s)",
    "s_clean = np.nan_to_num(s, nan=0.0)\n        fpr, tpr, _ = roc_curve(pos, s_clean)"
)
open(f, "w", encoding="utf-8").write(s)
print("fixed NaN in eval_controls")
print("done")