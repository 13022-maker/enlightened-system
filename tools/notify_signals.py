"""把 output/signals.json 中的新 BUY / SELL 訊號推播到 Telegram（沿用開明體系的 Bot 設定）。"""
import json
import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ICON = {"BUY": "🟢 買進", "SELL": "🔴 賣出"}


def main():
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    data = json.loads((ROOT / "output" / "signals.json").read_text(encoding="utf-8"))
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
    print(msg)
    if not (token and chat):
        print("（未設定 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID，略過推播）")
        return
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      data={"chat_id": chat, "text": msg[:4000], "parse_mode": "HTML"}, timeout=20)
    if not r.ok:
        print("Telegram 失敗：", r.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
