"""所有策略 walk-forward 比較報告。

用法：python tools/run_backtest.py [--strategies swing_momentum,dividend_yield] [--train-frac 0.6]
                                  [--no-point-in-time]
輸出：output/backtest_report.json、output/backtest_report.md
籌碼策略：data/chip/ 有真實資料才會納入；否則標註為模擬資料。

股票池：預設 point-in-time（data/universe/tw50_membership.csv 的歷史成分股，消除存活者偏差）；
       --no-point-in-time 改用 config.TW50 現行名單（舊行為，僅供比較）。
主表欄位：
    總報酬、CAGR（期間 < 1 年時不年化，等於總報酬並標註 *）、Sharpe ± 標準誤（Lo 2002 iid 簡化：
    SE = sqrt((1 + 0.5·SR_d²)/N)·sqrt(245)）、相對 0050 日超額報酬 t 值、參數組樣本外 Sharpe 中位數。
"""
import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quant.config import ETFS, OUTPUT_DIR, TW50  # noqa: E402
from quant.data import check_data_quality, load_panel  # noqa: E402
from quant.strategies import REGISTRY  # noqa: E402
from quant.validation import walk_forward  # noqa: E402

COLS = [("total_return", "總報酬", "%"), ("cagr", "年化", "cagr"), ("sharpe", "Sharpe±SE", "sr"),
        ("excess_t", "對0050 t值", "t"), ("max_drawdown", "最大回撤", "%"),
        ("annual_turnover", "年週轉", "f"), ("cost_drag", "成本拖累", "%"), ("win_rate", "勝率", "%"),
        ("n_trades", "交易數", "d")]


def fmt(m, key, kind):
    v = m.get(key)
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if kind == "%":
        return f"{v:+.1%}"
    if kind == "cagr":
        return f"{v:+.1%}" + ("" if m.get("annualized", True) else "*")
    if kind == "sr":
        return f"{v:.2f}±{m.get('sharpe_se', float('nan')):.2f}"
    if kind == "t":
        return f"{v:+.2f}"
    if kind == "f":
        return f"{v:.2f}"
    return str(v)


def row(name, m, median=None):
    cells = [fmt(m, k, t) for k, _, t in COLS]
    cells.append("—" if median is None else f"{median:.2f}")
    return "| " + name + " | " + " | ".join(cells) + " |"


def quality_section(panel, limit=40):
    issues = check_data_quality(panel)
    lines = ["## 資料品質檢查", ""]
    if not issues:
        return lines + ["- 無異常", ""], issues
    cnt = Counter(i["type"] for i in issues)
    label = {"gap_no_dividend": "開盤跳空<−3%（且比大盤再低3%）且無股利", "big_move": "單日含息漲跌>10.5%",
             "halt": "停牌／缺價格", "dividend_yield": "股利殖利率異常", "no_dividend_year": "整年無除息"}
    lines.append("、".join(f"{label.get(k, k)} {v} 筆" for k, v in cnt.items()))
    lines.append("")
    # 跳空筆數多，只列其他類型全部 + 跳空最多的股票
    for i in [x for x in issues if x["type"] != "gap_no_dividend"][:limit]:
        lines.append(f"- {i['code']} {i['date']} {label.get(i['type'], i['type'])}：{i['detail']}")
    top = Counter(i["code"] for i in issues if i["type"] == "gap_no_dividend").most_common(8)
    if top:
        lines.append("- 無股利大跳空最多：" + "、".join(f"{c}×{n}" for c, n in top)
                     + "（多為個股利空/題材股波動；若同日有除權息未記錄請補 data/dividends/）")
    lines.append("")
    return lines, issues


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default=",".join(REGISTRY))
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--no-point-in-time", action="store_true", help="改用現行 TW50 名單（有存活者偏差）")
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    args = ap.parse_args(argv)
    pit = not args.no_point_in_time

    panel = load_panel(list(TW50) + list(ETFS), chip=True, point_in_time=pit)
    results = {}
    for name in args.strategies.split(","):
        strat = REGISTRY[name]()
        print(f"回測 {name} ...", flush=True)
        wf = walk_forward(strat, panel, train_frac=args.train_frac)
        wf["title"] = strat.title
        wf["synthetic"] = bool(strat.needs_chip and panel.chip_is_synthetic)
        results[name] = wf

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = next(iter(results.values()))
    head = ("| 策略 | " + " | ".join(c[1] for c in COLS) + " | 參數組樣本外Sharpe中位數 |\n|"
            + "---|" * (len(COLS) + 2))
    test_days = first["test"]["n_days"]
    uni_txt = (f"point-in-time 歷史成分股（{int(panel.universe.any().sum())} 檔曾為成分股＋ETF）"
               if panel.universe is not None else "現行 TW50 名單（有存活者偏差）")
    lines = [f"# 策略回測比較（{datetime.now():%Y-%m-%d}）", "",
             f"- 資料：{panel.dates[0].date()} ~ {panel.dates[-1].date()}，{len(panel.codes)} 檔",
             f"- 股票池：{uni_txt}",
             f"- 訓練期：{first['train_period']}（挑參數）",
             f"- 測試期（樣本外）：{first['test_period']}，{test_days} 個交易日", ""]
    if not first["test"].get("annualized", True):
        lines += [f"> \\* 測試期 {test_days} 日 < 1 年（245 日），「年化」欄不年化、等於總報酬。", ""]
    lines += ["> Sharpe±SE：Lo (2002) iid 近似的 1 個標準誤；|t| < 2 代表與 0050 的差異在統計上無法區分。", "",
              "## 樣本外績效", "", head]
    for k, b in first["benchmarks_test"].items():
        lines.append(row(f"基準：{k}", b))
    for name, wf in results.items():
        tag = "（模擬籌碼，無參考價值）" if wf["synthetic"] else ""
        lines.append(row(f"{wf['title']}{tag}", wf["test"], wf["robustness"]["test_sharpe_median"]))
    lines += ["", "## 訓練期績效（挑參數用）", "", head]
    for name, wf in results.items():
        lines.append(row(wf["title"], wf["train"]))
    lines += ["", "## 參數穩健度（所有參數組的樣本外 Sharpe）", ""]
    for name, wf in results.items():
        r = wf["robustness"]
        lines.append(f"- {wf['title']}：中位數 {r['test_sharpe_median']:.2f}"
                     f"（{r['test_sharpe_min']:.2f} ~ {r['test_sharpe_max']:.2f}），採用參數 `{wf['best_params']}`")
    lines.append("")
    q_lines, issues = quality_section(panel)
    lines += q_lines
    if panel.universe is not None:
        lines += ["## 籌碼策略與股票池", "",
                  "- chip_flow 尚未套用 point-in-time 股票池：需在候選條件加上 `panel.universe_mask()`"
                  "（date×code 布林表，無成分股資料時全 True），與其他三個策略相同。", ""]
    (out_dir / "backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for wf in results.values():
        for g in wf["grid"]:
            g.pop("weights", None)
    payload = {**results}
    (out_dir / "backtest_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (out_dir / "data_quality.json").write_text(json.dumps(issues, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
