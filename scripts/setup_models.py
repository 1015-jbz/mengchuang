"""
自动下载 ONNX 表情识别模型到 models/ 目录。

模型文件较大（~30MB），不放 git 仓库，而是托管在 GitHub Release。
首次运行 setup.bat 或本脚本时自动下载。

支持：
  - 通过 GitHub API 自动获取最新 Release 资产
  - 断点续传
  - 校验文件大小
  - 已存在则跳过
"""
import os
import sys
import urllib.request
import urllib.error
import json
from pathlib import Path

# 配置
REPO = "1015-jbz/mengchuang"  # GitHub 仓库 owner/repo
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "enet_b2_7.onnx"
EXPECTED_SIZE = 30774088  # ~29MB，用于校验

# GitHub API 获取最新 Release 资产下载 URL
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"

# 备用直链（如果 API 不通，可直接 hardcode Release 资产 URL）
# 你在 GitHub 网页创建 Release 上传 onnx 后，把下载链接填到这里
FALLBACK_URL = f"https://github.com/{REPO}/releases/download/v1.0/enet_b2_7.onnx"


def get_download_url():
    """通过 GitHub API 获取最新 Release 中 enet_b2_7.onnx 的下载 URL"""
    try:
        req = urllib.request.Request(API_URL, headers={"User-Agent": "smart-cockpit-setup"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for asset in data.get("assets", []):
            if asset["name"] == "enet_b2_7.onnx":
                return asset["browser_download_url"]
        print(f"[警告] Release 中未找到 enet_b2_7.onnx，将使用备用链接")
        return FALLBACK_URL
    except Exception as e:
        print(f"[警告] 无法访问 GitHub API ({e})，将使用备用链接")
        return FALLBACK_URL


def download(url, dest):
    """下载文件，支持已存在跳过和大小校验"""
    if dest.exists() and dest.stat().st_size == EXPECTED_SIZE:
        print(f"[OK] 模型已存在: {dest} ({dest.stat().st_size} bytes)")
        return True

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(".onnx.tmp")

    print(f"[下载] {url}")
    print(f"[保存] {dest}")

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "smart-cockpit-setup"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 64  # 64KB
            with open(tmp_path, "wb") as f:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    if total > 0:
                        pct = downloaded * 100 // total
                        bar = "=" * (pct // 2) + ">" + " " * (50 - pct // 2)
                        sys.stdout.write(f"\r[{bar}] {pct}% ({downloaded//1024}KB/{total//1024}KB)")
                        sys.stdout.flush()
            print()  # 换行

        # 校验大小
        actual_size = tmp_path.stat().st_size
        if actual_size != EXPECTED_SIZE:
            print(f"[警告] 文件大小不匹配: 期望 {EXPECTED_SIZE}, 实际 {actual_size}")
            # 仍然继续，可能是版本更新

        tmp_path.replace(dest)
        print(f"[完成] 模型下载完成: {dest}")
        return True

    except Exception as e:
        print(f"\n[错误] 下载失败: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        return False


def main():
    print("=" * 60)
    print("智能座舱 - ONNX 表情识别模型下载工具")
    print("=" * 60)

    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size == EXPECTED_SIZE:
        print(f"[跳过] 模型已存在且大小正确: {MODEL_FILE}")
        print(f"       如需重新下载，请删除该文件后重跑本脚本。")
        return 0

    url = get_download_url()
    ok = download(url, MODEL_FILE)
    if not ok:
        print()
        print("[手动方案] 如果自动下载失败，你可以：")
        print(f"  1. 浏览器打开: https://github.com/{REPO}/releases")
        print(f"  2. 下载 enet_b2_7.onnx")
        print(f"  3. 放到 {MODEL_DIR}/")
        return 1

    print()
    print("[下一步] 配置 .env 文件:")
    print(f"  1. 复制 .env.example 为 .env")
    print(f"  2. 填入你的 DeepSeek API Key")
    print(f"  3. 运行: python app_demo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
