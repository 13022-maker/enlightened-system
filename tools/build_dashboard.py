"""把 output/signals.json 嵌入 dashboard/template.html，產出可直接開啟的 output/dashboard.html。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
data = json.loads((ROOT / "output" / "signals.json").read_text(encoding="utf-8"))
tpl = (ROOT / "dashboard" / "template.html").read_text(encoding="utf-8")
payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
html = tpl.replace("/*__DATA__*/null", payload)
body = html if html.lstrip().startswith("<!doctype") else (
    '<!doctype html>\n<html lang="zh-Hant"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body>\n' + html + "\n</body></html>")
(ROOT / "output" / "dashboard.html").write_text(body, encoding="utf-8")
(ROOT / "output" / "dashboard_artifact.html").write_text(html, encoding="utf-8")
print("✅ output/dashboard.html")
