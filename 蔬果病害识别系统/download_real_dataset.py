"""下载真实 PlantVillage 数据集并组织为训练/验证集

数据来源: https://github.com/spMohanty/PlantVillage-Dataset
(CC-BY 4.0, Mohanty et al. 2016, Frontiers in Plant Science)

从 raw/color/ 下选取 8 个类目 (苹果/葡萄/番茄 的健康与常见病害)，
分层 80/20 划分为训练集和验证集，保存到 data/train 和 data/val。
每类最多取 MAX_PER_CLASS 张以平衡类别、控制总规模。

用法:
    python download_real_dataset.py            # 下载并组织
    python download_real_dataset.py --clean   # 先清空旧数据再下载

依赖: requests (没有的话用 pip install requests)
"""

import os
import json
import time
import shutil
import argparse
import concurrent.futures
import urllib.request
import urllib.error
import urllib.parse

import config

# PlantVillage 文件夹名 -> 本项目类名
FOLDER_MAP = {
    "Apple___healthy":          "Apple_Healthy",
    "Apple___Apple_scab":       "Apple_Scab",
    "Apple___Cedar_apple_rust": "Apple_Rust",
    "Grape___healthy":          "Grape_Healthy",
    "Grape___Black_rot":        "Grape_BlackRot",
    "Tomato___healthy":         "Tomato_Healthy",
    "Tomato___Early_blight":    "Tomato_EarlyBlight",
    "Tomato___Late_blight":     "Tomato_LateBlight",
}

REPO = "spMohanty/PlantVillage-Dataset"
BRANCH = "master"
API_BASE = f"https://api.github.com/repos/{REPO}"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/raw/color"

MAX_PER_CLASS = 1000       # 每类最多下载数量 (PlantVillage 各类 152~5357 不等)
VAL_RATIO = 0.2            # 验证集比例
CONCURRENCY = 32           # 并发下载数
MAX_RETRIES = 3            # 单文件下载重试次数

USER_AGENT = "fruit-disease-project/1.0 (educational use)"


def api_get(url, retries=MAX_RETRIES):
    """带重试的 GitHub API GET (返回 JSON)。"""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 403 and "rate limit" in (e.read().decode("utf-8", "ignore")).lower():
                # API 限流，等待
                print("  [API 限流] 等待 30 秒后重试...")
                time.sleep(30)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"API 请求失败 {url}: {last_err}")


# 缓存: PlantVillage 文件夹名 -> git tree sha (从 raw/color 目录获取一次)
_FOLDER_SHA_CACHE = None


def _get_folder_sha_map():
    """获取 raw/color 下各子文件夹的 tree sha。

    contents API 返回目录列表 (本数据集共 38 个子目录，单页即可)，
    每个条目的 sha 即为该子目录的 git tree sha，可用于 git/trees API
    一次性获取该目录下全部文件 (无 1000 条分页限制)。
    """
    global _FOLDER_SHA_CACHE
    if _FOLDER_SHA_CACHE is not None:
        return _FOLDER_SHA_CACHE

    listing = api_get(f"{API_BASE}/contents/raw/color")
    sha_map = {}
    if isinstance(listing, list):
        for item in listing:
            if item.get("type") == "dir":
                sha_map[item["name"]] = item.get("sha")
    _FOLDER_SHA_CACHE = sha_map
    return sha_map


def list_folder_files(folder_name):
    """用 git/trees API 列出某文件夹下所有图片文件名。

    一次调用返回该 tree 的全部条目 (本数据集每类 < 5400 张，单次即可，
    不受 contents API 1000 条/页的限制)。
    """
    sha_map = _get_folder_sha_map()
    tree_sha = sha_map.get(folder_name)
    if not tree_sha:
        raise RuntimeError(f"找不到文件夹 {folder_name} 的 tree sha")

    tree = api_get(f"{API_BASE}/git/trees/{tree_sha}")
    files = []
    for item in tree.get("tree", []):
        if item.get("type") == "blob" and item["path"].lower().endswith((".jpg", ".jpeg", ".png")):
            files.append(item["path"])
    return files


def download_one(args):
    """下载单个文件。args=(raw_url, dest_path)"""
    raw_url, dest_path = args
    if os.path.exists(dest_path):
        return True, dest_path
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(raw_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(data)
            return True, dest_path
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return False, dest_path


def organize_class(folder_name, class_name, idx, total_classes):
    """下载并组织一个类别的图片到 train/val。"""
    print(f"\n[{idx}/{total_classes}] {folder_name} -> {class_name}")
    files = list_folder_files(folder_name)
    print(f"  GitHub 上共 {len(files)} 张图片")

    # 限制每类数量，随机采样
    import random
    rng = random.Random(config.RANDOM_SEED)
    if len(files) > MAX_PER_CLASS:
        files = rng.sample(files, MAX_PER_CLASS)
        print(f"  采样限制为 {MAX_PER_CLASS} 张")

    # 划分 train / val
    rng.shuffle(files)
    n_val = max(1, int(len(files) * VAL_RATIO))
    val_files = files[:n_val]
    train_files = files[n_val:]
    print(f"  划分: train={len(train_files)} val={len(val_files)}")

    # 构造下载任务 (文件名含空格/括号，需 URL 编码)
    tasks = []
    for fname in train_files:
        url = f"{RAW_BASE}/{folder_name}/{urllib.parse.quote(fname)}"
        dest = os.path.join(config.TRAIN_DIR, class_name, fname)
        tasks.append((url, dest))
    for fname in val_files:
        url = f"{RAW_BASE}/{folder_name}/{urllib.parse.quote(fname)}"
        dest = os.path.join(config.VAL_DIR, class_name, fname)
        tasks.append((url, dest))

    # 并发下载
    ok, fail = 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        results = list(pool.map(download_one, tasks))
    for success, _ in results:
        if success:
            ok += 1
        else:
            fail += 1
    print(f"  下载完成: 成功 {ok}, 失败 {fail}")
    return ok, fail


def clean_old_data():
    """清空旧的 (模拟) 数据。"""
    for d in [config.TRAIN_DIR, config.VAL_DIR, config.PREDICT_DIR]:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"已清空: {d}")


def make_predict_samples(n=8):
    """从验证集随机复制几张到 predict 目录供演示。"""
    os.makedirs(config.PREDICT_DIR, exist_ok=True)
    import random
    rng = random.Random(0)
    pool = []
    for class_name in config.CLASS_NAMES:
        d = os.path.join(config.VAL_DIR, class_name)
        if os.path.isdir(d):
            for f in os.listdir(d):
                pool.append(os.path.join(d, f))
    if not pool:
        return
    for f in rng.sample(pool, min(n, len(pool))):
        shutil.copy(f, os.path.join(config.PREDICT_DIR, os.path.basename(f)))
    print(f"\n预测样本已复制 {n} 张 -> {config.PREDICT_DIR}")


def main():
    parser = argparse.ArgumentParser(description="下载真实 PlantVillage 数据集")
    parser.add_argument("--clean", action="store_true", help="下载前清空 data/ 旧数据")
    args = parser.parse_args()

    print(f"数据来源: github.com/{REPO} (PlantVillage, CC-BY 4.0)")
    print(f"类别数: {len(FOLDER_MAP)}, 每类上限: {MAX_PER_CLASS}, 验证集比例: {VAL_RATIO}")
    print(f"并发数: {CONCURRENCY}")

    if args.clean:
        clean_old_data()

    os.makedirs(config.TRAIN_DIR, exist_ok=True)
    os.makedirs(config.VAL_DIR, exist_ok=True)

    total_ok = total_fail = 0
    t0 = time.time()
    for i, (pv_folder, class_name) in enumerate(FOLDER_MAP.items(), 1):
        ok, fail = organize_class(pv_folder, class_name, i, len(FOLDER_MAP))
        total_ok += ok
        total_fail += fail

    make_predict_samples()

    dt = time.time() - t0
    print(f"\n{'=' * 50}")
    print(f"下载完成! 总用时 {dt:.0f}s")
    print(f"成功 {total_ok} 张, 失败 {total_fail} 张")
    print(f"训练集: {config.TRAIN_DIR}")
    print(f"验证集: {config.VAL_DIR}")


if __name__ == "__main__":
    main()
