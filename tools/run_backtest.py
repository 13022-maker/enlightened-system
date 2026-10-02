"""所有策略 walk-forward 比較報告。

用法：python tools/run_backtest.py [--strategies swing_momentum,dividend_yield] [--train-frac 0.6]
輸出：output/backtest_report.json、output/backtest_report.md
籌碼策略：data/chip/ 有真實資料才會納入；否則標註為模擬資料。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.config import ETFS, OUTPUT_DIR, TW50  # noqa: E402
from quant.data import load_panel  # noqa: E402
from quant.strategies import REGISTRY  # noqa: E402
from quant.validation import walk_forward  # noqa: E402

COLS = [("cagr", "年化", "%"), ("sharpe", "Sharpe", "f"), ("max_drawdown", "最大回撤", "%"),
        ("annual_turnover", "年週轉", "f"), ("cost_drag", "成本拖累", "%"), ("win_rate", "勝率", "%"),
        ("n_trades", "交易數", "d")]


def fmt(v, kind):
    return f"{v:+.1%}" if kind == "%" else f"{v:.2f}" if kind == "f" else str(v)


def row(name, m):
    return "| " + name + " | " + " | ".join(fmt(m[k], t) for k, _, t in COLS) + " |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=",".join(REGISTRY))
    ap.add_argument("--train-frac", type=float, default=0.6)
    args = ap.parse_args()

    panel = load_panel(list(TW50) + list(ETFS), chip=True)
    results = {}
    for name in args.strategies.split(","):
        strat = REGISTRY[name]()
        print(f"回測 {name} ...", flush=True)
        wf = walk_forward(strat, panel, train_frac=args.train_frac)
        wf["title"] = strat.title
        wf["synthetic"] = bool(strat.needs_chip and panel.chip_is_synthetic)
        results[name] = wf

    OUTPUT_DIR.mkdir(exist_ok=True)
    first = next(iter(results.values()))
    head = "| 策略 | " + " | ".join(c[1] for c in COLS) + " |\n|" + "---|" * (len(COLS) + 1)
    lines = [f"# 策略回測比較（{datetime.now():%Y-%m-%d}）", "",
             f"- 資料：{panel.dates[0].date()} ~ {panel.dates[-1].date()}，{len(panel.codes)} 檔",
             f"- 訓練期：{first['train_period']}（挑參數）",
             f"- 測試期（樣本外）：{first['test_period']}", "",
             "## 樣本外績效", "", head]
    for k, b in first["benchmarks_test"].items():
        lines.append(row(f"基準：{k}", b))
    for name, wf in results.items():
        tag = "（模擬籌碼，無參考價值）" if wf["synthetic"] else ""
        lines.append(row(f"{wf['title']}{tag}", wf["test"]))
    lines += ["", "## 訓練期績效（挑參數用）", "", head]
    for name, wf in results.items():
        lines.append(row(wf["title"], wf["train"]))
    lines += ["", "## 參數穩健度（所有參數組的樣本外 Sharpe）", ""]
    for name, wf in results.items():
        r = wf["robustness"]
        lines.append(f"- {wf['title']}：中位數 {r['test_sharpe_median']:.2f}"
                     f"（{r['test_sharpe_min']:.2f} ~ {r['test_sharpe_max']:.2f}），採用參數 `{wf['best_params']}`")
    (OUTPUT_DIR / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for wf in results.values():
        for g in wf["grid"]:
            g.pop("weights", None)
    (OUTPUT_DIR / "backtest_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
