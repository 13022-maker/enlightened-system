"""把 output/signals.json 中的新 BUY / SELL 訊號推播到 Telegram（沿用開明體系的 Bot 設定）。

去重：只有 signals.json 的 as_of 與 output/last_notified.txt 不同時才推播，
      推播成功後寫入 as_of（假日 / 同日重跑不會重複推）。--force 可強制推播。
切割：訊息依「行」切成多則，每則 < 3500 字，不會切在 HTML 標籤中間。
失敗：推播失敗只印 ::warning::，離開碼仍為 0（不讓整個 workflow 變紅）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
ICON = {"BUY": "🟢 買進", "SELL": "🔴 賣出"}
MAX_CHARS = 3500


def build_message(data: dict) -> str:
    titles = {s["name"]: s["title"] for s in data["strategies"]}
    synthetic = {s["name"] for s in data["strategies"] if s["synthetic"]}
    lines = [f"📈 <b>開明體系 Quant 訊號</b> {data['as_of']}"]
    for name, title in titles.items():
        if name in synthetic:
            continue
        hits = [(s, s["signals"][name]) for s in data["stocks"]
                if name in s["signals"] and s["signals"][name]["action"] in ICON]
        if not hits:
            continue
        lines.append(f"\n<b>{title}</b>")
        for s, sig in hits:
            lines.append(f"{ICON[sig['action']]} {s['code']} {s['name']} {s['close']}｜{sig['reason']}")
    msg = "\n".join(lines) if len(lines) > 1 else lines[0] + "\n今日無新訊號"
    stale = data.get("stale_codes") or []
    if stale:
        msg += f"\n\n⚠️ 日K 未更新：{', '.join(stale)}"
    return msg


def split_message(msg: str, limit: int = MAX_CHARS) -> List[str]:
    """依行切割；每則長度 < limit。

    每一行本身的 HTML 標籤都是成對的（<b>...</b> 在同一行），所以只在換行處切就不會破壞標籤。
    單行超長時（理論上不會發生）先移除該行的 HTML 標籤再硬切，確保不會切在標籤中間。
    """
    import re

    chunks: List[str] = []
    cur = ""
    for line in msg.split("\n"):
        if len(line) >= limit:
            plain = re.sub(r"<[^>]+>", "", line)
            pieces = [plain[i:i + limit - 1] for i in range(0, len(plain), limit - 1)]
        else:
            pieces = [line]
        for piece in pieces:
            cand = piece if not cur else cur + "\n" + piece
            if len(cand) < limit:
                cur = cand
            else:
                if cur.strip():
                    chunks.append(cur)
                cur = piece
    if cur.strip():
        chunks.append(cur)
    return chunks


def send(token: str, chat: str, chunks: List[str]) -> bool:
    ok = True
    for i, text in enumerate(chunks, 1):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=20)
            if not r.ok:
                print(f"::warning::Telegram 推播失敗（第 {i}/{len(chunks)} 則）：{r.text[:300]}")
                ok = False
        except requests.RequestException as e:
            print(f"::warning::Telegram 推播例外（第 {i}/{len(chunks)} 則）：{e}")
            ok = False
    return ok


def main(argv: Optional[List[str]] = None, output_dir: Path = OUTPUT) -> int:
    ap = argparse.ArgumentParser(description="推播 Quant 訊號到 Telegram")
    ap.add_argument("--force", action="store_true", help="忽略 last_notified.txt 強制推播")
    a = ap.parse_args(argv)
    output_dir = Path(output_dir)
    sig_path, marker = output_dir / "signals.json", output_dir / "last_notified.txt"
    try:
        data = json.loads(sig_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"::warning::讀取 {sig_path} 失敗，略過推播：{e}")
        return 0
    as_of = str(data.get("as_of", ""))
    last = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if as_of and as_of == last and not a.force:
        print(f"as_of {as_of} 已推播過，略過（假日或同日重跑）")
        return 0

    msg = build_message(data)
    print(msg)
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        print("（未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，略過推播）")
        return 0
    if send(token, chat, split_message(msg)):
        marker.write_text(as_of + "\n", encoding="utf-8")   # 成功才記錄，失敗下次重跑會再推
    return 0


if __name__ == "__main__":
    sys.exit(main())
