"""開明體系 Quant 模組 — 全域設定。

TW50 是「近似的現行名單」（不是任何單一時點的真實成分股），只用來決定要抓哪些股票的資料。
回測時的歷史成分股（point-in-time）請用 data/universe/tw50_membership.csv（見 quant/universe.py），
否則會有存活者偏差（只用「現在」的贏家回測過去）。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY_DIR = ROOT / "data" / "daily"
CHIP_DIR = ROOT / "data" / "chip"
DIVIDEND_DIR = ROOT / "data" / "dividends"          # 除權息事件（覆寫/補齊日K 的 dividend 欄，見 quant/data.py）
UNIVERSE_FILE = ROOT / "data" / "universe" / "tw50_membership.csv"   # 歷史成分股（code,name,start,end）
OUTPUT_DIR = ROOT / "output"

TW50 = {
    "2330": "台積電", "2317": "鴻海", "2454": "聯發科", "2308": "台達電", "2382": "廣達",
    "2891": "中信金", "2881": "富邦金", "3711": "日月光投控", "2882": "國泰金", "2303": "聯電",
    "2412": "中華電", "2886": "兆豐金", "2884": "玉山金", "2357": "華碩", "2885": "元大金",
    "1216": "統一", "2892": "第一金", "3231": "緯創", "2345": "智邦", "2890": "永豐金",
    "5880": "合庫金", "2880": "華南金", "2327": "國巨", "3008": "大立光", "2887": "台新金",
    "2883": "凱基金", "6669": "緯穎", "2301": "光寶科", "1303": "南亞", "2002": "中鋼",
    "3045": "台灣大", "2207": "和泰車", "1101": "台泥", "4904": "遠傳", "2912": "統一超",
    "3034": "聯詠", "2603": "長榮", "2395": "研華", "5871": "中租-KY", "2379": "瑞昱",
    "1301": "台塑", "3037": "欣興", "2059": "川湖", "3017": "奇鋐", "2360": "致茂",
    "6505": "台塑化", "1326": "台化", "4938": "和碩", "2609": "陽明", "3653": "健策",
}
# 2024-02 之後曾是台灣50成分股、但不在上面 TW50 的股票（被調出或後來調入）。
# 回測需要它們的價格才能做 point-in-time 股票池（否則等於只回測「倖存者」）。
# 由 data/universe/tw50_membership.csv 整理；無 5 分K 資料者（6919 康霈*、7769 鴻勁）不列入。
EXTRA_CODES = {
    "9910": "豐泰", "2801": "彰銀", "2408": "南亞科", "1590": "亞德客-KY", "5876": "上海商銀",
    "3661": "世芯-KY", "6446": "藥華藥", "2615": "萬海", "2383": "台光電", "3665": "貿聯-KY",
    "2368": "金像電", "2449": "京元電子", "2344": "華邦電",
}
ETFS = {"0050": "元大台灣50", "0056": "元大高股息"}
BENCHMARK = "0050"


class Costs:
    """台股交易成本（以比例計）。"""
    FEE_RATE = 0.001425      # 券商手續費（買賣各一次）
    FEE_DISCOUNT = 0.6       # 網路下單常見 6 折，保守可設 1.0
    TAX_STOCK = 0.003        # 證交稅（賣出）
    TAX_ETF = 0.001          # ETF 證交稅（賣出）
    SLIPPAGE = 0.0005        # 單邊滑價假設 0.05%

    @classmethod
    def buy_cost(cls) -> float:
        return cls.FEE_RATE * cls.FEE_DISCOUNT + cls.SLIPPAGE

    @classmethod
    def sell_cost(cls, is_etf: bool = False) -> float:
        tax = cls.TAX_ETF if is_etf else cls.TAX_STOCK
        return cls.FEE_RATE * cls.FEE_DISCOUNT + tax + cls.SLIPPAGE


LIMIT_PCT = 0.10  # 漲跌停幅度（2015-06-01 起）
LIMIT_PCT_OLD = 0.07                        # 2015-06-01 以前上市股票漲跌幅 7%
LIMIT_CHANGE_DATE = pd.Timestamp("2015-06-01")
NEW_LISTING_FREE_DAYS = 5                   # 新股上市前 5 個交易日無漲跌幅限制


def limit_pct_on(dates) -> "pd.Series | float":
    """回傳各日期適用的漲跌停幅度（2015-06-01 前 7%，之後 10%）。dates 可為單一日期或 DatetimeIndex。"""
    if isinstance(dates, (pd.Timestamp, str)):
        return LIMIT_PCT_OLD if pd.Timestamp(dates) < LIMIT_CHANGE_DATE else LIMIT_PCT
    idx = pd.DatetimeIndex(dates)
    return pd.Series(((idx < LIMIT_CHANGE_DATE) * (LIMIT_PCT_OLD - LIMIT_PCT) + LIMIT_PCT), index=idx)
