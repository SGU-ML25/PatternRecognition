# %% [markdown]
# # SkyPose — Train model Pose Estimation trên Kaggle (chạy code từ GitHub)
#
# Notebook này **không chứa code model**: nó clone repo đồ án từ GitHub rồi gọi đúng các script
# trong `training/` (giống hệt khi chạy ở máy local):
#
# | Bước | Script trong repo |
# |---|---|
# | Cắt vùng người từ COCO | `training/prepare_data.py` |
# | Huấn luyện SimpleBaseline | `training/train.py` (+ `dataset.py`, `model.py`, `pose_utils.py`) |
# | Đánh giá COCO AP / PCK / tốc độ | `training/evaluate.py` |
# | Biểu đồ cho báo cáo | `training/make_figures.py` |
#
# ### Cài đặt trên Kaggle trước khi chạy
# 1. **Settings → Accelerator:** `GPU T4 x2` (hoặc `GPU P100`).
# 2. **Settings → Internet:** `On` (clone GitHub + tải trọng số ImageNet).
# 3. *(Khuyến nghị)* **Add Input** → dataset **COCO 2017** (vd. `awsaf49/coco-2017-dataset`).
#    Không gắn thì script tự tải ảnh từ `images.cocodataset.org` (chậm hơn).
# 4. Sửa `REPO_URL` ở ô cấu hình → **Save Version → Save & Run All (Commit)**.
#    Xong vào tab **Output** tải `out/skypose_<backbone>.zip`.
#
# **Repo private?** Tạo GitHub token (quyền đọc repo) → Kaggle **Add-ons → Secrets** thêm
# secret tên `GITHUB_TOKEN` và bật cho notebook này.

# %% [markdown]
# ## 1. Cấu hình

# %%
import os
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()

REPO_URL = os.environ.get("SKYPOSE_REPO_URL", "https://github.com/<username>/<repo>.git")  # ← sửa
BRANCH = "main"

BACKBONE = "resnet50"     # "resnet18" | "resnet34" | "resnet50"
EPOCHS = 20
BATCH_SIZE = 64           # tổng batch (DataParallel tự chia cho 2 GPU)
RESUME_FROM = None        # vd. "/kaggle/input/skypose-ckpt/last.pt" để train tiếp
SMOKE_TEST = not ON_KAGGLE  # chạy thử vài phút khi không ở trên Kaggle

BASE = Path("/kaggle/working") if ON_KAGGLE else Path(os.environ.get("SKYPOSE_SMOKE_DIR", "/tmp/skypose_smoke"))
REPO = BASE / "repo"                                     # mã nguồn clone về
OUT = BASE / "out"                                       # kết quả (lưu trong Output của Kaggle)
CROPS = Path("/tmp/skypose_crops") if ON_KAGGLE else BASE / "crops"   # crop tạm, không cần lưu
OUT.mkdir(parents=True, exist_ok=True)
if SMOKE_TEST:
    BACKBONE, EPOCHS, BATCH_SIZE = "resnet18", 1, 16
print(f"Kaggle={ON_KAGGLE} smoke={SMOKE_TEST} | {BACKBONE} {EPOCHS} epoch, batch {BATCH_SIZE}")

# %% [markdown]
# ## 2. Clone repo + cài thư viện

# %%
import shutil
import subprocess
import sys


def run(cmd, log_file=None, cwd=None):
    """Chạy lệnh, in output trực tiếp (và ghi vào log_file nếu có). Lỗi -> dừng notebook."""
    print("$", " ".join(map(str, cmd)), flush=True)
    log = open(log_file, "a") if log_file else None
    p = subprocess.Popen(list(map(str, cmd)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, cwd=cwd, env=os.environ.copy())
    for line in p.stdout:
        print(line, end="", flush=True)
        if log:
            log.write(line); log.flush()
    if p.wait() != 0:
        raise RuntimeError(f"Lệnh lỗi (exit {p.returncode}): {' '.join(map(str, cmd))}")


url = REPO_URL
if ON_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        url = REPO_URL.replace("https://", f"https://{token}@")
        print("Dùng GITHUB_TOKEN từ Kaggle Secrets (repo private)")
    except Exception:
        pass  # repo public
if REPO.exists():
    shutil.rmtree(REPO)
run(["git", "clone", "--depth", "1", "--branch", BRANCH, url, REPO])
run(["git", "-C", REPO, "log", "-1", "--format=commit %h  %s  (%cd)"])
run([sys.executable, "-m", "pip", "install", "-q", "-r", REPO / "requirements-train.txt"])
TRAIN = REPO / "training"

# %% [markdown]
# ## 3. Tìm dataset COCO-Keypoints 2017
# Đường dẫn được truyền cho code qua biến môi trường (xem `training/paths.py`):
# `SKYPOSE_ANN_DIR` (annotation) và `SKYPOSE_CROPS` (nơi ghi crop).

# %%
import urllib.request
import zipfile


def find_coco(root):
    """Quét /kaggle/input (giới hạn độ sâu, không đi vào thư mục ảnh) tìm annotation + ảnh."""
    ann = train_img = val_img = None
    if not root.exists():
        return ann, train_img, val_img
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        if len(d.relative_to(root).parts) > 6:
            dirnames[:] = []
            continue
        if "person_keypoints_train2017.json" in filenames and ann is None:
            ann = d
        for name in ("train2017", "val2017"):
            if name in dirnames and next((d / name).glob("*.jpg"), None) is not None:
                if name == "train2017" and train_img is None:
                    train_img = d / name
                if name == "val2017" and val_img is None:
                    val_img = d / name
        dirnames[:] = [x for x in dirnames if x not in ("train2017", "val2017", "test2017", "unlabeled2017")]
    return ann, train_img, val_img


search_root = Path("/kaggle/input") if ON_KAGGLE else Path(os.environ.get("SKYPOSE_LOCAL_COCO", "data/coco")).resolve()
ANN_DIR, TRAIN_IMAGES, VAL_IMAGES = find_coco(search_root)
if ANN_DIR is None:
    zpath = BASE / "ann.zip"
    print("Không thấy annotation -> tải từ cocodataset.org")
    urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", zpath)
    with zipfile.ZipFile(zpath) as zf:
        for n in zf.namelist():
            if "person_keypoints" in n:
                zf.extract(n, BASE)
    zpath.unlink()
    ANN_DIR = BASE / "annotations"

os.environ["SKYPOSE_ANN_DIR"] = str(ANN_DIR)
os.environ["SKYPOSE_CROPS"] = str(CROPS)
print("annotation :", ANN_DIR)
print("train2017  :", TRAIN_IMAGES or "(không có -> tải lẻ từng ảnh)")
print("val2017    :", VAL_IMAGES or "(không có -> tải lẻ từng ảnh)")
print("crops      :", CROPS)

# %% [markdown]
# ## 4. Chuẩn bị dữ liệu — `training/prepare_data.py`

# %%
for split, images in (("val", VAL_IMAGES), ("train", TRAIN_IMAGES)):
    if (CROPS / f"{split}_index.npz").exists():
        print(f"{split}: đã có, bỏ qua")
        continue
    cmd = [sys.executable, TRAIN / "prepare_data.py", "--split", split,
           "--workers", 16 if images else 64]
    if images:
        cmd += ["--images-dir", images]
    if SMOKE_TEST:
        cmd += ["--limit-images", 150 if split == "train" else 200]
    run(cmd)

# %% [markdown]
# ## 5. Huấn luyện — `training/train.py`
# Output của lệnh được ghi vào `out/train_<backbone>.log` (dùng để vẽ đường loss).

# %%
CKPT_DIR = OUT / BACKBONE
TRAIN_LOG = OUT / f"train_{BACKBONE}.log"
cmd = [sys.executable, TRAIN / "train.py", "--backbone", BACKBONE, "--epochs", EPOCHS,
       "--bs", BATCH_SIZE, "--out", CKPT_DIR]
if RESUME_FROM:
    cmd += ["--resume-from", RESUME_FROM]
elif (CKPT_DIR / "last.pt").exists():
    cmd += ["--resume"]
if SMOKE_TEST:
    cmd += ["--limit", 320, "--eval-limit", 200]
run(cmd, log_file=TRAIN_LOG)

# %% [markdown]
# ## 6. Đánh giá cuối trên `best.pt` — `training/evaluate.py`

# %%
EVAL_JSON = OUT / f"eval_{BACKBONE}.json"
run([sys.executable, TRAIN / "evaluate.py", "--ckpt", CKPT_DIR / "best.pt", "--out", EVAL_JSON])

# %% [markdown]
# ## 7. Biểu đồ — `training/make_figures.py`

# %%
FIG_DIR = OUT / "figures"
cmd = [sys.executable, TRAIN / "make_figures.py",
       "--run", f"{BACKBONE}:{TRAIN_LOG}:{CKPT_DIR / 'train_log.csv'}",
       "--eval", EVAL_JSON, "--fig-dir", FIG_DIR, "--ckpt", CKPT_DIR / "best.pt"]
if VAL_IMAGES:
    cmd += ["--val-images", VAL_IMAGES]
else:
    cmd = cmd[:cmd.index("--ckpt")]  # không có ảnh val gốc -> bỏ ảnh minh hoạ
run(cmd, cwd=REPO)

try:
    from IPython.display import Image, display
    for f in sorted(FIG_DIR.iterdir()):
        print(f.name)
        display(Image(filename=str(f), width=720))
except ImportError:
    pass

# %% [markdown]
# ## 8. Đóng gói kết quả
# `skypose_<backbone>.zip` gồm `best.pt` (dùng cho server game), log, `train_log.csv`,
# `eval_<backbone>.json` và các biểu đồ. `last.pt` (kèm optimizer) để riêng trong `out/<backbone>/`
# để train tiếp nếu cần.
#
# Ở máy local: giải nén vào `checkpoints/<backbone>/` rồi chạy
# `SKYPOSE_CKPT=checkpoints/<backbone>/best.pt .venv/bin/uvicorn server.app:app --port 8765`.

# %%
zip_path = OUT / f"skypose_{BACKBONE}.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(CKPT_DIR / "best.pt", "best.pt")
    z.write(CKPT_DIR / "train_log.csv", "train_log.csv")
    for p in (TRAIN_LOG, EVAL_JSON):
        z.write(p, p.name)
    for p in FIG_DIR.iterdir():
        z.write(p, f"figures/{p.name}")
print(f"{zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
if ON_KAGGLE:
    shutil.rmtree(REPO, ignore_errors=True)  # không cần lưu mã nguồn trong Output
