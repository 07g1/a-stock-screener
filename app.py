#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
  A股涨停板筛选器 — 一进二 + 二进三 Streamlit 交互式应用
=====================================================================
数据来源: AKShare (东方财富网公开接口)
运行要求:
    pip install akshare pandas streamlit plotly
    需要联网, 建议交易日 9:25 竞价结束后运行

两种模式:
    模式一 · 一进二  -> 筛选首板种子选手, 五维评分评估二板潜力
                        ①涨停时间与强度 ②封单质量 ③量能与换手
                        ④炸板次数 ⑤人气热度与板块效应
    模式二 · 二进三  -> 筛选2连板标的, 四维评分评估三板潜力
                        ①题材强度 ②量能形态(前缩后放/前放后缩)
                        ③位置形态 ④资金面与龙虎榜

用法:
    streamlit run zt_erjin3_screener.py
    然后在浏览器中打开 http://localhost:8501

注意:
    - 建议在交易日 9:25 后运行 (竞价结束, 数据已更新)
    - 首次加载会拉取较多数据, 请耐心等待
    - 历史K线逐股获取, 候选股票较多时耗时较长
=====================================================================
"""

import logging
import re
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ============================ 页面配置 ============================
# 必须是第一个 st.* 调用
st.set_page_config(
    page_title="A股涨停板筛选器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 抑制 Streamlit 模块加载阶段的 "missing ScriptRunContext" 无害警告
class _SuppressScriptRunContextWarning(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()

try:
    _target_logger = logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context")
    _target_logger.addFilter(_SuppressScriptRunContextWarning())
except Exception:
    pass  # 日志路径可能在 Streamlit 版本间变化，忽略

try:
    import akshare as ak
except ImportError:
    st.error("缺少依赖, 请先执行: pip install akshare pandas streamlit plotly")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ============================ 配置区 ============================
class Config:
    # ==================== 共用阈值 ====================
    MAX_PRICE = 30.0                # 最高股价(元)
    MAX_FLOAT_MCAP_YI = 200.0       # 最大流通市值(亿元)
    MA_PERIODS = [5, 10, 20]        # 均线周期
    MA_LOOKBACK = 60                # 均线计算回溯天数

    # ---- 排除板块 ----
    EXCLUDE_PREFIXES = ("8", "4")   # 北交所(8开头), 新三板(4开头)
    EXCLUDE_BOARDS = ("688",)       # 科创板
    EXCLUDE_ST = True               # 排除ST

    TOP_N = 30                      # 默认展示数量

    # ==================== 一进二: 减法排除 ====================
    OVERHEATED_RECENT_COUNT = 5     # "涨停统计"近N个交易日内出现次数>=该值, 视为前期爆炒
    CHECK_HALF_YEAR_GENE = False    # 半年涨停基因校验(逐股调历史K线, 较慢)
    MIN_HALF_YEAR_ZT = 3            # 半年内涨停次数下限

    # ==================== 一进二: 板块效应 ====================
    CHECK_SECTOR_EFFECT = True      # 联网核查板块效应
    SECTOR_MIN_PEERS_UP5 = 5        # 同板块涨幅>=5%个股数
    SECTOR_PEER_UP_PCT = 5.0        # 涨幅阈值

    # ==================== 一进二: 五维评分权重 (满分100) ====================
    W_TIME = 30      # ① 涨停时间与强度
    W_SEAL = 25      # ② 封单质量
    W_VOL = 20       # ③ 量能与换手
    W_ZB = 15        # ④ 炸板次数
    W_SECTOR = 10    # ⑤ 人气热度与板块效应

    # ==================== 二进三: 题材筛选 ====================
    SECTOR_MIN_ZT_COUNT = 3         # 板块至少N只涨停股才视为"主线"
    SECTOR_CORE_ZT_RANK = 3         # 板块内封板时间排名前N视为"核心股"

    # ==================== 二进三: 量能形态 ====================
    VOL_SHRINK_RATIO = 0.85         # 当日量/前日量 < 0.85 视为缩量
    VOL_EXPAND_RATIO = 1.15         # 当日量/前日量 > 1.15 视为放量
    TURNOVER_LOCKUP_MAX = 8.0       # 锁仓阶段换手率上限(%)

    # ==================== 二进三: 位置与形态 ====================
    MAX_CONSECUTIVE_HISTORY = 4     # 近期有过N连板以上的视为高位, 回避
    HIGH_OPEN_THRESHOLD = 7.0       # 三板高开阈值(%)

    # ==================== 二进三: 资金面 ====================
    LHB_TOP5_BUY_MAX_PCT = 20.0     # 龙虎榜前五买入占比上限(%)

    # ==================== 二进三: 四维评分权重 (满分100) ====================
    W_THEME = 30       # 题材强度
    W_VOLUME = 30      # 量能形态
    W_POSITION = 25    # 位置形态
    W_CAPITAL = 15     # 资金面


# ============================ 通用工具函数 ============================
@st.cache_data(ttl=300, show_spinner=False)
def get_col(df, candidates):
    """按候选列名列表在df.columns中查找(先精确匹配, 再子串兜底)"""
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        for c in candidates:
            if c in str(col):
                return col
    return None


def to_float(x, default=None):
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except (ValueError, TypeError):
        return default


def parse_time(t_str):
    """兼容多种封板时间格式: '093000' / '09:30:00' / '09:30'"""
    if t_str is None or (isinstance(t_str, float) and pd.isna(t_str)):
        return None
    t_str = str(t_str).strip()
    if t_str.replace(".", "").isdigit() and ":" not in t_str:
        t_str = t_str.split(".")[0].zfill(6)
        t_str = f"{t_str[0:2]}:{t_str[2:4]}:{t_str[4:6]}"
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(t_str, fmt).time()
        except ValueError:
            continue
    return None


def parse_zt_stat(stat_str):
    """解析"涨停统计"字段, 形如 '3/10' 表示近10个交易日里涨停3次"""
    if stat_str is None or (isinstance(stat_str, float) and pd.isna(stat_str)):
        return None, None
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*", str(stat_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def is_stock_excluded(code: str) -> Tuple[bool, str]:
    """检查股票是否应被排除"""
    code = str(code).zfill(6)
    if Config.EXCLUDE_ST and "ST" in code.upper():
        return True, "ST股"
    if code.startswith(Config.EXCLUDE_PREFIXES):
        return True, "北交所/新三板"
    if code.startswith(Config.EXCLUDE_BOARDS):
        return True, "科创板"
    return False, ""


# ============================ 数据获取 (带缓存) ============================
@st.cache_data(ttl=300, show_spinner="正在获取涨停股池...")
def fetch_zt_pool(date_str: str) -> pd.DataFrame:
    """获取当日涨停股池"""
    try:
        df = ak.stock_zt_pool_em(date=date_str)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"获取涨停池数据失败: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner="正在获取全市场快照...")
def fetch_spot_market() -> Optional[pd.DataFrame]:
    """获取全市场实时行情快照 (用于补充振幅/量比)"""
    try:
        return ak.stock_zh_a_spot_em()
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_history(code: str, days: int = 60) -> Optional[pd.DataFrame]:
    """获取单只股票历史K线"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
        if df is None or df.empty:
            return None
        return df.tail(days).copy()
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner="正在获取龙虎榜数据...")
def fetch_lhb_data(date_str: str) -> Optional[pd.DataFrame]:
    """获取指定日期的龙虎榜明细"""
    try:
        df = ak.stock_lhb_detail_em(date=date_str)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sector_peers_up5(industry_name: str, cache: dict) -> Optional[int]:
    """给定所属行业名称, 返回该板块当日涨幅>=5%的个股数量(联网, 带缓存与限速)"""
    if not industry_name:
        return None
    if industry_name in cache:
        return cache[industry_name]
    try:
        cons = ak.stock_board_industry_cons_em(symbol=industry_name)
        pct_col = get_col(cons, ["涨跌幅"])
        count = int((cons[pct_col] >= Config.SECTOR_PEER_UP_PCT).sum()) if pct_col else None
    except Exception:
        count = None
    cache[industry_name] = count
    time.sleep(0.25)
    return count


def count_half_year_zt(code: str) -> Optional[int]:
    """逐股回溯近约半年(125个交易日)涨跌幅, 统计涨停次数(用于'缺乏涨停基因'硬性校验)"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="")
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.tail(125)
    pct_col = get_col(df, ["涨跌幅"])
    if pct_col is None:
        return None
    if code.startswith(("300", "301", "688")):
        threshold = 19.5
    else:
        threshold = 9.5
    return int((df[pct_col] >= threshold).sum())


# ========================================================================
#                       一进二 模块
# ========================================================================

def pre_filter_yijiner(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    一进二减法排除:
    1. 排除 ST / 北交所 / 科创板
    2. 排除高价股 / 大市值股
    3. 排除前期爆炒股 (涨停统计>=N)
    4. 可选: 半年涨停基因校验
    返回: (kept_df, dropped_df)
    """
    col_price = get_col(df, ["最新价"])
    col_mcap = get_col(df, ["流通市值"])
    col_stat = get_col(df, ["涨停统计"])
    col_industry = get_col(df, ["所属行业"])
    col_code = get_col(df, ["代码"])
    col_name = get_col(df, ["名称"])

    industry_zt_count = df[col_industry].value_counts().to_dict() if col_industry else {}

    kept_rows, dropped_rows = [], []
    for _, row in df.iterrows():
        reasons = []
        code = str(row.get(col_code, "")).zfill(6)

        excluded, reason = is_stock_excluded(code)
        if excluded:
            reasons.append(f"板块排除: {reason}")

        price = to_float(row.get(col_price)) if col_price else None
        if price is not None and price > Config.MAX_PRICE:
            reasons.append(f"高价股: {price}元 > {Config.MAX_PRICE}元")

        mcap = to_float(row.get(col_mcap)) if col_mcap else None
        if mcap is not None and mcap > Config.MAX_FLOAT_MCAP_YI * 1e8:
            reasons.append(f"大市值: {mcap/1e8:.1f}亿 > {Config.MAX_FLOAT_MCAP_YI}亿")

        cnt, window = parse_zt_stat(row.get(col_stat)) if col_stat else (None, None)
        if cnt is not None and cnt >= Config.OVERHEATED_RECENT_COUNT:
            reasons.append(f"前期爆炒: 近{window}个交易日涨停{cnt}次")

        if Config.CHECK_HALF_YEAR_GENE and col_code:
            zt_half_year = count_half_year_zt(code)
            if zt_half_year is not None and zt_half_year < Config.MIN_HALF_YEAR_ZT:
                reasons.append(f"缺乏涨停基因: 半年内涨停约{zt_half_year}次 < {Config.MIN_HALF_YEAR_ZT}次")

        soft_flag = ""
        if col_industry:
            industry = row.get(col_industry)
            if industry and industry_zt_count.get(industry, 0) < 2:
                soft_flag = "板块效应待观察(同行业当日涨停家数<2)"

        record = row.to_dict()
        record["_drop_reasons"] = "; ".join(reasons)
        record["_soft_flag"] = soft_flag
        record["_sector_zt_count"] = industry_zt_count.get(row.get(col_industry), 0) if col_industry and row.get(col_industry) else 0

        (dropped_rows if reasons else kept_rows).append(record)

    kept_df = pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame()
    dropped_df = pd.DataFrame(dropped_rows) if dropped_rows else pd.DataFrame()
    return kept_df, dropped_df


# ---- 一进二五维评分函数 ----

def score_time_strength_yijiner(row, col_first_seal, col_amplitude) -> Tuple[float, str]:
    """① 涨停时间与强度 (满分 W_TIME)"""
    amp = to_float(row.get(col_amplitude)) if col_amplitude else None
    t = parse_time(row.get(col_first_seal)) if col_first_seal else None

    if t is None:
        return 0.0, "无封板时间数据"

    # 一字板检测: 振幅极小意味着筹码未充分换手, 接力资金难入场
    if amp is not None and amp < 1.0:
        return 0.0, f"疑似一字板(全天振幅仅{amp:.2f}%, 筹码未充分换手, 后续接力资金难入场)"

    thresholds = [
        (datetime.strptime("09:49:00", "%H:%M:%S").time(), 1.00, "9:49前封板,最优质"),
        (datetime.strptime("10:30:00", "%H:%M:%S").time(), 0.80, "10:30前封板,优质"),
        (datetime.strptime("11:30:00", "%H:%M:%S").time(), 0.55, "上午封板,尚可"),
        (datetime.strptime("13:30:00", "%H:%M:%S").time(), 0.35, "午后早段封板,一般"),
        (datetime.strptime("14:30:00", "%H:%M:%S").time(), 0.15, "14:30前封板,偏弱"),
    ]
    for limit, ratio, note in thresholds:
        if t <= limit:
            return round(Config.W_TIME * ratio, 1), f"{note}(首封{t.strftime('%H:%M:%S')}, 振幅{amp if amp is not None else 'N/A'})"
    return 0.0, f"14:30后尾盘偷袭板(首封{t.strftime('%H:%M:%S')}), 连板概率极低"


def score_seal_quality_yijiner(row, col_seal_amt, col_turnover_amt) -> Tuple[float, str]:
    """② 封单质量 (满分 W_SEAL)"""
    seal = to_float(row.get(col_seal_amt)) if col_seal_amt else None
    turn_amt = to_float(row.get(col_turnover_amt)) if col_turnover_amt else None
    if not seal or not turn_amt:
        return 0.0, "封板资金/成交额数据缺失"
    ratio = seal / turn_amt
    if ratio < 0.05:
        return round(Config.W_SEAL * 0.30, 1), f"封板勉强(封单/成交额={ratio:.1%})"
    elif ratio < 0.10:
        return round(Config.W_SEAL * 0.70, 1), f"强势(封单/成交额={ratio:.1%})"
    elif ratio <= 0.20:
        return round(Config.W_SEAL * 1.00, 1), f"极度强势,潜在龙头(封单/成交额={ratio:.1%})"
    else:
        return round(Config.W_SEAL * 0.85, 1), f"封单占比过高,关注流动性({ratio:.1%})"


def score_volume_turnover_yijiner(row, col_turnover_rate, col_volume_ratio) -> Tuple[float, str]:
    """③ 量能与换手 (满分 W_VOL)"""
    tr = to_float(row.get(col_turnover_rate)) if col_turnover_rate else None
    vr = to_float(row.get(col_volume_ratio)) if col_volume_ratio else None

    if tr is None:
        tr_score, tr_note = 0.5, "换手率数据缺失"
    elif 5 <= tr <= 20:
        tr_score, tr_note = 1.0, f"换手率{tr:.1f}%健康"
    elif tr > 30:
        tr_score, tr_note = 0.10, f"换手率{tr:.1f}%过高,警惕潜伏盘出货"
    elif tr < 5:
        tr_score, tr_note = 0.5, f"换手率{tr:.1f}%偏低"
    else:
        tr_score, tr_note = 0.6, f"换手率{tr:.1f}%尚可"

    if vr is None:
        vr_score, vr_note = 0.5, "量比数据缺失"
    elif 1.5 <= vr <= 3:
        vr_score, vr_note = 1.0, f"量比{vr:.1f}健康放量"
    elif vr > 5:
        vr_score, vr_note = 0.10, f"量比{vr:.1f}巨量,警惕"
    elif vr < 1.5:
        vr_score, vr_note = 0.4, f"量比{vr:.1f}量能不足"
    else:
        vr_score, vr_note = 0.6, f"量比{vr:.1f}尚可"

    score = round(Config.W_VOL * (0.5 * tr_score + 0.5 * vr_score), 1)
    return score, f"{tr_note}; {vr_note}"


def score_zb_yijiner(row, col_zb) -> Tuple[float, str]:
    """④ 炸板次数 (满分 W_ZB)"""
    zb = to_float(row.get(col_zb)) if col_zb else 0
    zb = zb or 0
    if zb == 0:
        return float(Config.W_ZB), "未炸板,主力做多意愿坚定"
    elif zb == 1:
        return round(Config.W_ZB * 0.40, 1), "炸板1次,需观察回封是否二次堆量"
    else:
        return 0.0, f"反复炸板{int(zb)}次,主力做多意愿不坚定"


def score_sector_effect_yijiner(row, col_industry, df_today, sector_cache) -> Tuple[float, str]:
    """⑤ 人气热度与板块效应 (满分 W_SECTOR)"""
    industry = row.get(col_industry) if col_industry else None
    if not industry:
        return 0.0, "无行业/板块数据"

    zt_count_in_sector = int((df_today[col_industry] == industry).sum())
    peers_up5 = fetch_sector_peers_up5(industry, sector_cache) if Config.CHECK_SECTOR_EFFECT else None

    hot = zt_count_in_sector >= 10
    has_effect = (peers_up5 is not None and peers_up5 >= Config.SECTOR_MIN_PEERS_UP5) or zt_count_in_sector >= 5

    if hot:
        score = float(Config.W_SECTOR)
    elif has_effect:
        score = Config.W_SECTOR * 0.7
    elif zt_count_in_sector >= 2:
        score = Config.W_SECTOR * 0.4
    else:
        score = 0.0

    note = f"板块[{industry}]当日涨停{zt_count_in_sector}家"
    if peers_up5 is not None:
        note += f", 涨幅>=5%的同板块个股{peers_up5}只"
    return round(score, 1), note


def analyze_single_yijiner(code: str, name: str, row: dict, zt_df: pd.DataFrame,
                           sector_cache: dict) -> dict:
    """对单只首板股票进行五维评分"""
    code = str(code).zfill(6)

    col_first_seal = get_col(zt_df, ["首次封板时间"])
    col_amplitude = get_col(zt_df, ["振幅"])
    col_seal_amt = get_col(zt_df, ["封板资金", "封单金额", "封单资金"])
    col_turnover_amt = get_col(zt_df, ["成交额"])
    col_turnover_rate = get_col(zt_df, ["换手率"])
    col_volume_ratio = get_col(zt_df, ["量比"])
    col_zb = get_col(zt_df, ["炸板次数"])
    col_industry = get_col(zt_df, ["所属行业"])
    col_price = get_col(zt_df, ["最新价"])
    col_mcap = get_col(zt_df, ["流通市值"])

    s_time, n_time = score_time_strength_yijiner(row, col_first_seal, col_amplitude)
    s_seal, n_seal = score_seal_quality_yijiner(row, col_seal_amt, col_turnover_amt)
    s_vol, n_vol = score_volume_turnover_yijiner(row, col_turnover_rate, col_volume_ratio)
    s_zb, n_zb = score_zb_yijiner(row, col_zb)
    s_sector, n_sector = score_sector_effect_yijiner(row, col_industry, zt_df, sector_cache)

    total = round(s_time + s_seal + s_vol + s_zb + s_sector, 1)

    price = to_float(row.get(col_price)) if col_price else None
    float_mcap = to_float(row.get(col_mcap)) if col_mcap else None

    return {
        "代码": code,
        "名称": name,
        "总分": total,
        "行业": row.get(col_industry) if col_industry else None,
        "最新价": price,
        "流通市值_亿": round(float_mcap / 1e8, 1) if float_mcap else None,
        "①时间强度": s_time,
        "①备注": n_time,
        "②封单质量": s_seal,
        "②备注": n_seal,
        "③量能换手": s_vol,
        "③备注": n_vol,
        "④炸板次数": s_zb,
        "④备注": n_zb,
        "⑤板块效应": s_sector,
        "⑤备注": n_sector,
        "软标记": row.get("_soft_flag", ""),
    }


# ========================================================================
#                       二进三 模块
# ========================================================================

def step1_erjin3_exclude_and_theme(zt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    二进三第一步过滤:
    1. 排除 ST / 北交所 / 科创板
    2. 排除非2连板
    3. 排除高价股 / 大市值股
    4. 标记板块效应
    """
    if zt_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    col_lianban = get_col(zt_df, ["连板数"])
    col_code = get_col(zt_df, ["代码"])
    col_price = get_col(zt_df, ["最新价"])
    col_mcap = get_col(zt_df, ["流通市值"])
    col_industry = get_col(zt_df, ["所属行业"])

    if col_lianban:
        zt_df = zt_df[zt_df[col_lianban].fillna(0).astype(float) == 2].copy()
    else:
        st.warning("未找到'连板数'列, 无法筛选2连板个股")

    if zt_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if col_industry:
        industry_zt_count = zt_df[col_industry].value_counts().to_dict()
    else:
        industry_zt_count = {}

    kept_rows, dropped_rows = [], []
    for _, row in zt_df.iterrows():
        reasons = []
        code = str(row.get(col_code, "")).zfill(6)

        excluded, reason = is_stock_excluded(code)
        if excluded:
            reasons.append(f"板块排除: {reason}")

        price = to_float(row.get(col_price)) if col_price else None
        if price is not None and price > Config.MAX_PRICE:
            reasons.append(f"高价股: {price}元 > {Config.MAX_PRICE}元")

        mcap = to_float(row.get(col_mcap)) if col_mcap else None
        if mcap is not None and mcap > Config.MAX_FLOAT_MCAP_YI * 1e8:
            reasons.append(f"大市值: {mcap/1e8:.1f}亿 > {Config.MAX_FLOAT_MCAP_YI}亿")

        industry = row.get(col_industry) if col_industry else None
        sector_zt_count = industry_zt_count.get(industry, 0) if industry else 0
        soft_flag = ""
        if sector_zt_count < Config.SECTOR_MIN_ZT_COUNT:
            soft_flag = f"板块效应不足(同板块涨停{sector_zt_count}家 < {Config.SECTOR_MIN_ZT_COUNT}家)"

        record = row.to_dict()
        record["_drop_reasons"] = "; ".join(reasons)
        record["_soft_flag"] = soft_flag
        record["_sector_zt_count"] = sector_zt_count

        (dropped_rows if reasons else kept_rows).append(record)

    kept_df = pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame()
    dropped_df = pd.DataFrame(dropped_rows) if dropped_rows else pd.DataFrame()
    return kept_df, dropped_df


def analyze_volume_pattern_erjin3(code: str, hist_df: pd.DataFrame) -> dict:
    """二进三: 分析前两板的量能形态"""
    if hist_df is None or len(hist_df) < 4:
        return {
            "pattern": "数据不足", "advice": "无法判断", "score_ratio": 0.4,
            "turnover_rates": [], "vol_ratios": [],
            "detail": "历史K线数据不足, 无法分析量能形态",
        }

    pct_col = get_col(hist_df, ["涨跌幅"])
    vol_col = get_col(hist_df, ["成交量"])
    turnover_col = get_col(hist_df, ["换手率"])

    if pct_col is None or vol_col is None:
        return {
            "pattern": "数据缺失", "advice": "无法判断", "score_ratio": 0.4,
            "turnover_rates": [], "vol_ratios": [],
            "detail": "涨跌幅/成交量字段缺失",
        }

    hist_df = hist_df.copy()
    hist_df["_pct"] = hist_df[pct_col].apply(to_float, default=0)
    hist_df["_vol"] = hist_df[vol_col].apply(to_float, default=0)
    if turnover_col:
        hist_df["_turnover"] = hist_df[turnover_col].apply(to_float, default=None)

    if code.startswith(("300", "301")):
        zt_threshold = 19.5
    else:
        zt_threshold = 9.5

    zt_days = hist_df[hist_df["_pct"] >= zt_threshold].tail(3)
    if len(zt_days) < 2:
        return {
            "pattern": "涨停日不足", "advice": "无法判断", "score_ratio": 0.3,
            "turnover_rates": [], "vol_ratios": [],
            "detail": f"近{len(hist_df)}个交易日仅找到{len(zt_days)}个涨停日",
        }

    board1 = zt_days.iloc[-2]
    board2 = zt_days.iloc[-1]

    vol1 = board1["_vol"]
    vol2 = board2["_vol"]

    b1_idx = hist_df.index.get_loc(board1.name)
    vol0 = hist_df.iloc[b1_idx - 1]["_vol"] if b1_idx > 0 else vol1

    ratio_b1 = vol1 / vol0 if vol0 > 0 else 1.0
    ratio_b2 = vol2 / vol1 if vol1 > 0 else 1.0

    def vol_label(ratio):
        if ratio < Config.VOL_SHRINK_RATIO:
            return "缩量"
        elif ratio > Config.VOL_EXPAND_RATIO:
            return "放量"
        return "平量"

    label_b1 = vol_label(ratio_b1)
    label_b2 = vol_label(ratio_b2)
    pattern = f"{label_b1}→{label_b2}"

    turnover_rates = []
    if turnover_col:
        t1 = board1.get("_turnover")
        t2 = board2.get("_turnover")
        if t1 is not None:
            turnover_rates.append(round(t1, 2))
        if t2 is not None:
            turnover_rates.append(round(t2, 2))

    if label_b1 == "缩量" and label_b2 == "缩量":
        advice = "前两板缩量(惜售), 三板需放量消化获利盘 → 关注今日是否放量上攻"
        score_ratio = 0.65
    elif label_b1 == "放量" and label_b2 == "放量":
        t2_val = turnover_rates[-1] if turnover_rates else 999
        if t2_val <= Config.TURNOVER_LOCKUP_MAX:
            advice = f"前两板放量后二板换手{t2_val}%已进入锁仓阶段, 三板有望缩量加速"
            score_ratio = 0.90
        else:
            advice = f"前两板放量但二板换手{t2_val}%偏高, 三板需缩量确认(换手<{Config.TURNOVER_LOCKUP_MAX}%)"
            score_ratio = 0.60
    elif label_b1 == "放量" and label_b2 == "缩量":
        advice = "放量→缩量, 分歧转一致, 筹码锁定良好, 三板有望加速"
        score_ratio = 0.95
    elif label_b1 == "缩量" and label_b2 == "放量":
        advice = "缩量→放量, 二板分歧加大, 三板需继续放量或缩量确认"
        score_ratio = 0.70
    else:
        advice = f"量能平缓({label_b1}→{label_b2}), 需观察三板方向选择"
        score_ratio = 0.55

    return {
        "pattern": pattern, "advice": advice, "score_ratio": score_ratio,
        "turnover_rates": turnover_rates,
        "vol_ratios": [round(ratio_b1, 2), round(ratio_b2, 2)],
        "detail": f"一板量比={ratio_b1:.2f}({label_b1}), 二板量比={ratio_b2:.2f}({label_b2})",
    }


def analyze_position_erjin3(code: str, hist_df: pd.DataFrame, price: float,
                            float_mcap_yi: float) -> dict:
    """二进三: 分析位置与形态"""
    result = {
        "price_ok": price <= Config.MAX_PRICE,
        "mcap_ok": float_mcap_yi <= Config.MAX_FLOAT_MCAP_YI,
        "ma_bullish": False, "ma_detail": "",
        "overheated": False, "overheat_detail": "",
        "score_ratio": 0.5,
    }

    if hist_df is None or len(hist_df) < Config.MA_LOOKBACK:
        result["ma_detail"] = "K线数据不足, 无法计算均线"
        return result

    close_col = get_col(hist_df, ["收盘"])
    if close_col is None:
        result["ma_detail"] = "收盘价字段缺失"
        return result

    hist_df = hist_df.copy()
    hist_df["_close"] = hist_df[close_col].apply(to_float, default=0)
    closes = hist_df["_close"].values

    ma_values = {}
    ma_ok = []
    for period in Config.MA_PERIODS:
        if len(closes) >= period:
            ma = pd.Series(closes).rolling(period).mean().iloc[-1]
            ma_values[period] = round(ma, 2)
            if closes[-1] > ma:
                ma_ok.append(period)

    result["ma_values"] = ma_values
    result["ma_bullish"] = len(ma_ok) == len(Config.MA_PERIODS)
    if ma_values:
        ma_str = ", ".join([f"MA{p}={v}" for p, v in ma_values.items()])
        result["ma_detail"] = f"收盘{closes[-1]:.2f}, {ma_str}, 站上均线: {ma_ok}"
    else:
        result["ma_detail"] = "均线数据不足"

    pct_col = get_col(hist_df, ["涨跌幅"])
    if pct_col:
        hist_df["_pct"] = hist_df[pct_col].apply(to_float, default=0)
        zt_threshold = 19.5 if code.startswith(("300", "301")) else 9.5
        consecutive = 0
        max_consecutive = 0
        for pct in hist_df["_pct"].values:
            if pct >= zt_threshold:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        if max_consecutive >= Config.MAX_CONSECUTIVE_HISTORY:
            result["overheated"] = True
            result["overheat_detail"] = f"近期出现过{max_consecutive}连板, 抛压沉重"
        else:
            result["overheat_detail"] = f"近期最高{max_consecutive}连板, 无过度炒作风险"

    score = 0.5
    if result["price_ok"]:
        score += 0.15
    if result["mcap_ok"]:
        score += 0.15
    if result["ma_bullish"]:
        score += 0.15
    if not result["overheated"]:
        score += 0.05
    result["score_ratio"] = min(score, 1.0)

    return result


def analyze_capital_flow_erjin3(code: str, lhb_df: Optional[pd.DataFrame]) -> dict:
    """二进三: 分析龙虎榜资金面"""
    result = {
        "has_lhb_data": False, "top5_buy_pct": None, "top5_buy_ok": True,
        "net_buy": None, "detail": "", "score_ratio": 0.5,
    }

    if lhb_df is None or lhb_df.empty:
        result["detail"] = "无龙虎榜数据 (当日未上榜)"
        result["score_ratio"] = 0.6
        return result

    code_col = get_col(lhb_df, ["代码"])
    if code_col is None:
        result["detail"] = "龙虎榜数据格式异常"
        result["score_ratio"] = 0.5
        return result

    stock_lhb = lhb_df[lhb_df[code_col].astype(str).str.zfill(6) == code.zfill(6)]
    if stock_lhb.empty:
        result["detail"] = "该股未上龙虎榜"
        result["score_ratio"] = 0.6
        return result

    result["has_lhb_data"] = True
    row = stock_lhb.iloc[0]

    turnover_col = get_col(stock_lhb, ["成交额"])
    turnover = to_float(row.get(turnover_col)) if turnover_col else None

    buy_cols = []
    for c in stock_lhb.columns:
        if "买入" in str(c) and ("金额" in str(c) or "席位" in str(c)):
            buy_cols.append(c)
    if not buy_cols:
        buy_total_col = get_col(stock_lhb, ["买入总计", "买方合计"])
        if buy_total_col:
            buy_total = to_float(row.get(buy_total_col))
            if buy_total and turnover:
                result["top5_buy_pct"] = round(buy_total / turnover * 100, 1)
    else:
        buy_sum = sum(to_float(row.get(c), 0) for c in buy_cols)
        if turnover and turnover > 0:
            result["top5_buy_pct"] = round(buy_sum / turnover * 100, 1)

    net_buy_col = get_col(stock_lhb, ["净买额", "净买入"])
    if net_buy_col:
        result["net_buy"] = to_float(row.get(net_buy_col))

    if result["top5_buy_pct"] is not None:
        if result["top5_buy_pct"] <= Config.LHB_TOP5_BUY_MAX_PCT:
            result["top5_buy_ok"] = True
            result["score_ratio"] = 0.85
        else:
            result["top5_buy_ok"] = False
            result["score_ratio"] = 0.30
            result["detail"] = f"前五买入占比{result['top5_buy_pct']}% > {Config.LHB_TOP5_BUY_MAX_PCT}%, 次日砸盘风险大"
    else:
        result["score_ratio"] = 0.6

    if result["net_buy"] is not None:
        if result["net_buy"] > 0:
            result["detail"] += f", 主力净买入{result['net_buy']/1e4:.0f}万"
        else:
            result["detail"] += f", 主力净卖出{abs(result['net_buy'])/1e4:.0f}万"
            result["score_ratio"] = max(result["score_ratio"] - 0.15, 0.1)

    if not result["detail"]:
        result["detail"] = "龙虎榜数据正常"

    return result


def score_stock_erjin3(row: dict, theme_result: dict, volume_result: dict,
                       position_result: dict, capital_result: dict) -> dict:
    """二进三: 综合四维评分, 满分100"""
    sector_zt = theme_result.get("sector_zt_count", 0)
    is_core = theme_result.get("is_core", False)

    if sector_zt >= 10:
        theme_score = Config.W_THEME * 1.0
        theme_note = f"板块涨停{sector_zt}家, 人气极旺"
    elif sector_zt >= 5:
        theme_score = Config.W_THEME * 0.85
        theme_note = f"板块涨停{sector_zt}家, 主线明确"
    elif sector_zt >= Config.SECTOR_MIN_ZT_COUNT:
        theme_score = Config.W_THEME * 0.65
        theme_note = f"板块涨停{sector_zt}家, 初具板块效应"
    elif sector_zt >= 2:
        theme_score = Config.W_THEME * 0.40
        theme_note = f"板块涨停仅{sector_zt}家, 板块效应较弱"
    else:
        theme_score = Config.W_THEME * 0.15
        theme_note = f"板块涨停仅{sector_zt}家, 单打独斗"

    if is_core:
        theme_score += 3
        theme_note += " | 板块核心股(封板早/封单大)"

    theme_score = min(round(theme_score, 1), Config.W_THEME)

    vol_score = round(Config.W_VOLUME * volume_result.get("score_ratio", 0.4), 1)
    vol_note = f"{volume_result.get('pattern', 'N/A')} | {volume_result.get('advice', '')}"

    pos_score = round(Config.W_POSITION * position_result.get("score_ratio", 0.5), 1)
    issues = []
    if position_result.get("overheated"):
        issues.append("近期过度炒作")
    if not position_result.get("ma_bullish"):
        issues.append("均线非多头排列")
    if not position_result.get("price_ok"):
        issues.append("股价偏高")
    if not position_result.get("mcap_ok"):
        issues.append("市值偏大")
    pos_note = position_result.get("ma_detail", "")
    if issues:
        pos_note += f" | ⚠️ {'; '.join(issues)}"

    cap_score = round(Config.W_CAPITAL * capital_result.get("score_ratio", 0.5), 1)
    cap_note = capital_result.get("detail", "")

    total = round(theme_score + vol_score + pos_score + cap_score, 1)

    return {
        "总分": total,
        "①题材强度": theme_score, "①备注": theme_note,
        "②量能形态": vol_score, "②备注": vol_note,
        "③位置形态": pos_score, "③备注": pos_note,
        "④资金面": cap_score, "④备注": cap_note,
    }


def analyze_single_erjin3(code: str, name: str, row: dict, zt_df: pd.DataFrame,
                          lhb_df: Optional[pd.DataFrame]) -> Optional[dict]:
    """二进三: 对单个2连板股票进行完整四维分析"""
    code = str(code).zfill(6)

    hist_df = fetch_stock_history(code, Config.MA_LOOKBACK + 10)

    col_industry = get_col(zt_df, ["所属行业"])
    col_first_seal = get_col(zt_df, ["首次封板时间"])
    col_price = get_col(zt_df, ["最新价"])
    col_mcap = get_col(zt_df, ["流通市值"])

    industry = row.get(col_industry) if col_industry else None
    if industry:
        sector_zt_count = int((zt_df[col_industry] == industry).sum())
    else:
        sector_zt_count = 0

    is_core = False
    if col_first_seal and col_industry:
        sector_rows = zt_df[zt_df[col_industry] == industry]
        times = []
        for _, sr in sector_rows.iterrows():
            t = parse_time(sr.get(col_first_seal))
            if t:
                times.append((sr.get(get_col(zt_df, ["代码"])), t))
        times.sort(key=lambda x: x[1])
        core_codes = [t[0] for t in times[:Config.SECTOR_CORE_ZT_RANK]]
        if code in [str(c).zfill(6) for c in core_codes]:
            is_core = True

    theme_result = {
        "sector_zt_count": sector_zt_count,
        "is_core": is_core,
        "industry": industry,
    }

    volume_result = analyze_volume_pattern_erjin3(code, hist_df)

    price = to_float(row.get(col_price), 999) if col_price else 999
    float_mcap_yi = to_float(row.get(col_mcap), 99999) / 1e8 if col_mcap else 99999
    position_result = analyze_position_erjin3(code, hist_df, price, float_mcap_yi)

    capital_result = analyze_capital_flow_erjin3(code, lhb_df)

    score = score_stock_erjin3(row, theme_result, volume_result, position_result, capital_result)

    return {
        "代码": code, "名称": name, "行业": industry,
        "最新价": price if price != 999 else None,
        "流通市值_亿": round(float_mcap_yi, 1) if float_mcap_yi != 99999 else None,
        **score,
        "量能模式": volume_result.get("pattern", ""),
        "二板换手": volume_result.get("turnover_rates", [None, None])[-1] if volume_result.get("turnover_rates") else None,
        "均线多头": "✓" if position_result.get("ma_bullish") else "✗",
        "过度炒作": "⚠" if position_result.get("overheated") else "✓",
        "板块核心": "✓" if is_core else "",
        "软标记": row.get("_soft_flag", ""),
    }


# ========================================================================
#                       Streamlit UI — 侧边栏
# ========================================================================

def render_sidebar() -> Tuple[str, str, dict]:
    """渲染侧边栏, 返回 (模式, 日期字符串, 配置覆盖字典)"""
    st.sidebar.title("📈 涨停板筛选器")

    # ---- 模式选择 ----
    mode = st.sidebar.radio(
        "🎯 分析模式",
        ["一进二 (首板→二板)", "二进三 (二板→三板)"],
        key="analysis_mode",
        help="一进二: 筛选首板种子选手, 五维评分评估二板潜力\n二进三: 筛选2连板标的, 四维评分评估三板潜力"
    )
    is_yijiner = "一进二" in mode

    st.sidebar.caption(
        "首板→二板候选评估" if is_yijiner else "2连板→3连板候选评估"
    )

    # ---- 日期选择 ----
    today = datetime.now()
    date_input = st.sidebar.date_input(
        "选择交易日",
        value=today,
        format="YYYY-MM-DD",
        help="建议选择最近的交易日, 收盘后数据最完整"
    )
    date_str = date_input.strftime("%Y%m%d")

    st.sidebar.divider()

    # ---- 共用阈值 ----
    st.sidebar.subheader("⚙️ 筛选阈值")

    with st.sidebar.expander("📋 基础过滤", expanded=False):
        max_price = st.number_input("最高股价(元)", 10.0, 100.0, Config.MAX_PRICE, 5.0, key="cfg_max_price")
        max_mcap = st.number_input("最大流通市值(亿)", 50.0, 500.0, Config.MAX_FLOAT_MCAP_YI, 50.0, key="cfg_max_mcap")

    # ---- 模式特有阈值 ----
    if is_yijiner:
        with st.sidebar.expander("🎯 一进二: 减法排除", expanded=False):
            overheated = st.number_input("前期爆炒阈值(涨停统计>=N次)", 2, 15, Config.OVERHEATED_RECENT_COUNT, key="cfg_overheated")
            check_gene = st.checkbox("开启半年涨停基因校验(较慢)", Config.CHECK_HALF_YEAR_GENE, key="cfg_check_gene")
            min_gene = st.number_input("半年最少涨停次数", 1, 10, Config.MIN_HALF_YEAR_ZT, key="cfg_min_gene") if check_gene else Config.MIN_HALF_YEAR_ZT

        with st.sidebar.expander("🔥 一进二: 板块效应", expanded=False):
            check_sector = st.checkbox("联网核查板块效应(略耗时)", Config.CHECK_SECTOR_EFFECT, key="cfg_check_sector")
            sector_peers = st.number_input("同板块涨幅≥5%最少个股数", 2, 20, Config.SECTOR_MIN_PEERS_UP5, key="cfg_sector_peers")

        st.sidebar.caption(
            f"评分权重: 时间{Config.W_TIME} + 封单{Config.W_SEAL} + "
            f"量能{Config.W_VOL} + 炸板{Config.W_ZB} + 板块{Config.W_SECTOR} = 100"
        )
    else:
        with st.sidebar.expander("🎯 二进三: 题材筛选", expanded=False):
            sector_min = st.number_input("板块最少涨停家数", 2, 20, Config.SECTOR_MIN_ZT_COUNT, key="cfg_sector_min")

        with st.sidebar.expander("📊 二进三: 量能形态", expanded=False):
            shrink_ratio = st.slider("缩量判定阈值", 0.5, 1.0, Config.VOL_SHRINK_RATIO, 0.05, key="cfg_shrink")
            expand_ratio = st.slider("放量判定阈值", 1.0, 2.0, Config.VOL_EXPAND_RATIO, 0.05, key="cfg_expand")
            lockup_turnover = st.slider("锁仓换手率上限(%)", 3.0, 15.0, Config.TURNOVER_LOCKUP_MAX, 0.5, key="cfg_lockup")

        with st.sidebar.expander("💰 二进三: 资金面", expanded=False):
            lhb_pct = st.slider("龙虎榜买入占比上限(%)", 10.0, 40.0, Config.LHB_TOP5_BUY_MAX_PCT, 5.0, key="cfg_lhb")

        st.sidebar.caption(
            f"评分权重: 题材{Config.W_THEME} + 量能{Config.W_VOLUME} + "
            f"位置{Config.W_POSITION} + 资金{Config.W_CAPITAL} = 100"
        )

    # ---- 排除设置 ----
    with st.sidebar.expander("🚫 排除设置", expanded=False):
        exclude_st = st.checkbox("排除ST股", Config.EXCLUDE_ST, key="cfg_exclude_st")
        exclude_kechuang = st.checkbox("排除科创板(688)", True, key="cfg_exclude_kc")
        exclude_beijiao = st.checkbox("排除北交所(8开头)", True, key="cfg_exclude_bj")

    st.sidebar.divider()

    top_n = st.sidebar.number_input("展示数量上限", 0, 100, Config.TOP_N, 5, help="0=展示全部")

    # 构建配置覆盖
    config_overrides = {
        "MAX_PRICE": max_price,
        "MAX_FLOAT_MCAP_YI": max_mcap,
        "EXCLUDE_ST": exclude_st,
        "TOP_N": top_n,
    }

    if is_yijiner:
        config_overrides.update({
            "OVERHEATED_RECENT_COUNT": overheated,
            "CHECK_HALF_YEAR_GENE": check_gene,
            "MIN_HALF_YEAR_ZT": min_gene if check_gene else Config.MIN_HALF_YEAR_ZT,
            "CHECK_SECTOR_EFFECT": check_sector,
            "SECTOR_MIN_PEERS_UP5": sector_peers,
        })
    else:
        config_overrides.update({
            "SECTOR_MIN_ZT_COUNT": sector_min,
            "VOL_SHRINK_RATIO": shrink_ratio,
            "VOL_EXPAND_RATIO": expand_ratio,
            "TURNOVER_LOCKUP_MAX": lockup_turnover,
            "LHB_TOP5_BUY_MAX_PCT": lhb_pct,
        })

    if exclude_kechuang:
        config_overrides["EXCLUDE_BOARDS"] = ("688",)
    else:
        config_overrides["EXCLUDE_BOARDS"] = ()
    if exclude_beijiao:
        config_overrides["EXCLUDE_PREFIXES"] = ("8", "4")
    else:
        config_overrides["EXCLUDE_PREFIXES"] = ("4",)

    return mode, date_str, config_overrides


def apply_config_overrides(overrides: dict):
    """应用配置覆盖"""
    for k, v in overrides.items():
        if hasattr(Config, k):
            setattr(Config, k, v)


# ========================================================================
#                       Streamlit UI — 通用渲染
# ========================================================================

def color_score(val):
    """总分颜色映射"""
    if val >= 75:
        return "background-color: #d4edda; color: #155724"
    elif val >= 60:
        return "background-color: #fff3cd; color: #856404"
    elif val >= 45:
        return "background-color: #f8d7da; color: #721c24"
    return ""


def render_summary_metrics(results: list):
    """渲染统计指标行"""
    if not results:
        return
    col1, col2, col3, col4 = st.columns(4)
    scores = [r["总分"] for r in results]
    col1.metric("候选数量", len(results))
    col2.metric("平均分", f"{sum(scores)/len(scores):.1f}" if scores else "N/A")
    col3.metric("最高分", f"{max(scores):.1f}" if scores else "N/A")
    col4.metric("≥75分(优质)", sum(1 for s in scores if s >= 75))


def render_sector_analysis(zt_df: pd.DataFrame):
    """渲染板块热度分析 (两种模式共用)"""
    if zt_df.empty:
        return

    st.subheader("🔥 板块热度分析")

    col_industry = get_col(zt_df, ["所属行业"])
    if not col_industry:
        st.info("无板块数据")
        return

    sector_counts = zt_df[col_industry].value_counts().head(20)

    if HAS_PLOTLY:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                x=sector_counts.values,
                y=sector_counts.index,
                orientation="h",
                title="板块涨停家数 Top 20",
                labels={"x": "涨停家数", "y": "板块"},
            )
            fig.update_layout(height=500, yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top5 = sector_counts.head(5)
            others = sector_counts.iloc[5:].sum()
            pie_data = pd.DataFrame({
                "板块": list(top5.index) + (["其他"] if others > 0 else []),
                "涨停家数": list(top5.values) + ([others] if others > 0 else []),
            })
            fig2 = px.pie(pie_data, values="涨停家数", names="板块", title="涨停分布(前5板块)")
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.dataframe(sector_counts.rename("涨停家数"), use_container_width=True)

    st.caption("**主线板块**(涨停≥3家):")
    main_sectors = sector_counts[sector_counts >= 3]
    if len(main_sectors) > 0:
        cols = st.columns(min(len(main_sectors), 5))
        for i, (sector, count) in enumerate(main_sectors.items()):
            with cols[i % 5]:
                st.metric(sector, f"{count}家涨停")
    else:
        st.warning("当前无满足条件的板块主线")


# ========================================================================
#                       Streamlit UI — 一进二渲染
# ========================================================================

def render_results_table_yijiner(results: list, top_n: int):
    """一进二结果表格"""
    if not results:
        st.info("没有符合条件的一进二候选标的")
        return

    results = sorted(results, key=lambda x: x.get("总分", 0), reverse=True)
    if top_n > 0:
        results = results[:top_n]

    display_cols = [
        "代码", "名称", "总分", "行业",
        "①时间强度", "②封单质量", "③量能换手", "④炸板次数", "⑤板块效应",
        "最新价", "流通市值_亿",
    ]

    df_display = pd.DataFrame(results)
    available_cols = [c for c in display_cols if c in df_display.columns]
    df_display = df_display[available_cols]

    styled = df_display.style.map(
        color_score, subset=["总分"] if "总分" in df_display.columns else []
    )

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(35 * len(df_display) + 38, 800),
        hide_index=True,
    )

    render_summary_metrics(results)
    return df_display


def render_stock_detail_yijiner(results: list):
    """一进二个股详情 (五维柱状图)"""
    if not results:
        return

    st.subheader("🔍 个股深度分析")

    stock_options = {f"{r['代码']} {r['名称']}": r for r in results}
    selected = st.selectbox(
        "选择股票查看详情",
        list(stock_options.keys()),
        key="detail_select_yijiner"
    )

    if not selected:
        return

    r = stock_options[selected]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总分", f"{r['总分']}/100")
    with col2:
        st.metric("行业", r.get("行业", "N/A"))
    with col3:
        st.metric("最新价", f"{r.get('最新价', 'N/A')}元")

    if HAS_PLOTLY:
        dimensions = ["①时间强度", "②封单质量", "③量能换手", "④炸板次数", "⑤板块效应"]
        max_scores = [Config.W_TIME, Config.W_SEAL, Config.W_VOL, Config.W_ZB, Config.W_SECTOR]
        values = [r.get(d, 0) for d in dimensions]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="得分",
            x=dimensions,
            y=values,
            text=[f"{v}/{m}" for v, m in zip(values, max_scores)],
            textposition="outside",
            marker_color=["#28a745" if v/m >= 0.7 else "#ffc107" if v/m >= 0.4 else "#dc3545"
                          for v, m in zip(values, max_scores)],
        ))
        fig.add_trace(go.Bar(
            name="满分",
            x=dimensions,
            y=max_scores,
            marker_color="rgba(0,0,0,0.1)",
            showlegend=False,
        ))
        fig.update_layout(
            barmode="overlay",
            height=350,
            margin=dict(t=10, b=10),
            yaxis=dict(range=[0, max(max_scores) + 5]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("**评分明细:**")
    for dim, label in [("①", "时间强度"), ("②", "封单质量"), ("③", "量能换手"), ("④", "炸板次数"), ("⑤", "板块效应")]:
        note = r.get(f"{dim}备注", "")
        if note:
            st.caption(f"**{label}**: {note}")

    soft_flag = r.get("软标记", "")
    if soft_flag:
        st.warning(f"⚠️ {soft_flag}")


def render_methodology_yijiner():
    """一进二方法论文档"""
    st.subheader("📖 一进二方法论")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "① 时间强度", "② 封单质量", "③ 量能换手", "④ 炸板次数", "⑤ 板块效应"
    ])

    with tab1:
        st.markdown("""
        ### ① 涨停时间与强度 (满分30分)

        封板时间越早, 主力做多意愿越强, 二板概率越高。

        | 封板时间 | 得分比例 | 评价 |
        |---|---|---|
        | 9:49 前 | 100% | 最优质, 强势封板 |
        | 10:30 前 | 80% | 优质, 早盘封板 |
        | 11:30 前 | 55% | 尚可, 上午封板 |
        | 13:30 前 | 35% | 一般, 午后早段 |
        | 14:30 前 | 15% | 偏弱, 下午封板 |
        | 14:30 后 | 0% | 尾盘偷袭板, 连板概率极低 |

        > **一字板特殊处理**: 若全天振幅<1%, 视为一字板, 不给分(筹码未换手, 接力资金难入场)
        """)

    with tab2:
        st.markdown("""
        ### ② 封单质量 (满分25分)

        封单金额/成交额比值反映封板决心:

        | 封单/成交额 | 得分 | 评价 |
        |---|---|---|
        | 10%~20% | 100% | 极度强势, 潜在龙头 |
        | 5%~10% | 70% | 强势封板 |
        | <5% | 30% | 封板勉强 |
        | >20% | 85% | 封单占比过高, 关注流动性 |

        > 封单过小说明主力信心不足; 封单过大可能是一日游资金
        """)

    with tab3:
        st.markdown("""
        ### ③ 量能与换手 (满分20分)

        **换手率** (占50%权重):
        - 5%~20%: 健康换手, 满分
        - <5%: 偏低, 50%
        - 20%~30%: 尚可, 60%
        - >30%: 过高, 警惕潜伏盘出货, 10%

        **量比** (占50%权重):
        - 1.5~3: 健康放量, 满分
        - <1.5: 量能不足, 40%
        - 3~5: 尚可, 60%
        - >5: 巨量, 警惕, 10%
        """)

    with tab4:
        st.markdown("""
        ### ④ 炸板次数 (满分15分)

        | 炸板次数 | 得分 | 评价 |
        |---|---|---|
        | 0次 | 100% | 未炸板, 主力做多意愿坚定 |
        | 1次 | 40% | 炸板1次, 需观察回封是否二次堆量 |
        | ≥2次 | 0% | 反复炸板, 主力做多意愿不坚定 |

        > 炸板后回封的股票, 需关注回封时的量能变化和封单强度
        """)

    with tab5:
        st.markdown("""
        ### ⑤ 人气热度与板块效应 (满分10分)

        | 条件 | 得分 |
        |---|---|
        | 板块当日涨停≥10家 | 100% (人气极旺) |
        | 板块有明确效应(涨幅≥5%个股≥5只) | 70% |
        | 板块涨停≥2家 | 40% |
        | 单打独斗(板块涨停<2家) | 0% |

        > **板块效应是一进二成功的关键**: 有板块效应的个股, 二板成功率远高于单打独斗的个股
        > 单打独斗的个股很难走远。若同题材内有竞争股存在, 需警惕卡位风险
        """)


# ========================================================================
#                       Streamlit UI — 二进三渲染
# ========================================================================

def render_results_table_erjin3(results: list, top_n: int):
    """二进三结果表格"""
    if not results:
        st.info("没有符合条件的二进三候选标的")
        return

    results = sorted(results, key=lambda x: x.get("总分", 0), reverse=True)
    if top_n > 0:
        results = results[:top_n]

    display_cols = [
        "代码", "名称", "总分", "行业",
        "①题材强度", "②量能形态", "③位置形态", "④资金面",
        "量能模式", "二板换手", "均线多头", "过度炒作", "板块核心",
        "最新价", "流通市值_亿",
    ]

    df_display = pd.DataFrame(results)
    available_cols = [c for c in display_cols if c in df_display.columns]
    df_display = df_display[available_cols]

    styled = df_display.style.map(
        color_score, subset=["总分"] if "总分" in df_display.columns else []
    )

    st.dataframe(
        styled,
        use_container_width=True,
        height=min(35 * len(df_display) + 38, 800),
        hide_index=True,
    )

    render_summary_metrics(results)
    return df_display


def render_stock_detail_erjin3(results: list):
    """二进三个股详情 (四维柱状图)"""
    if not results:
        return

    st.subheader("🔍 个股深度分析")

    stock_options = {f"{r['代码']} {r['名称']}": r for r in results}
    selected = st.selectbox(
        "选择股票查看详情",
        list(stock_options.keys()),
        key="detail_select_erjin3"
    )

    if not selected:
        return

    r = stock_options[selected]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总分", f"{r['总分']}/100")
    with col2:
        st.metric("行业", r.get("行业", "N/A"))
    with col3:
        st.metric("量能模式", r.get("量能模式", "N/A"))

    if HAS_PLOTLY:
        dimensions = ["①题材强度", "②量能形态", "③位置形态", "④资金面"]
        max_scores = [Config.W_THEME, Config.W_VOLUME, Config.W_POSITION, Config.W_CAPITAL]
        values = [r.get(d, 0) for d in dimensions]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="得分",
            x=dimensions,
            y=values,
            text=[f"{v}/{m}" for v, m in zip(values, max_scores)],
            textposition="outside",
            marker_color=["#28a745" if v/m >= 0.7 else "#ffc107" if v/m >= 0.4 else "#dc3545"
                          for v, m in zip(values, max_scores)],
        ))
        fig.add_trace(go.Bar(
            name="满分",
            x=dimensions,
            y=max_scores,
            marker_color="rgba(0,0,0,0.1)",
            showlegend=False,
        ))
        fig.update_layout(
            barmode="overlay",
            height=350,
            margin=dict(t=10, b=10),
            yaxis=dict(range=[0, max(max_scores) + 5]),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("**评分明细:**")
    for dim, label in [("①", "题材强度"), ("②", "量能形态"), ("③", "位置形态"), ("④", "资金面")]:
        note = r.get(f"{dim}备注", "")
        if note:
            st.caption(f"**{label}**: {note}")

    soft_flag = r.get("软标记", "")
    if soft_flag:
        st.warning(f"⚠️ {soft_flag}")


def render_methodology_erjin3():
    """二进三方法论文档"""
    st.subheader("📖 二进三方法论")

    tab1, tab2, tab3, tab4 = st.tabs([
        "① 题材筛选", "② 量能配合", "③ 位置形态", "④ 排除与审查"
    ])

    with tab1:
        st.markdown("""
        ### 第一步: 题材筛选 — 只做主线, 不做杂毛

        - **强逻辑支撑**: 有明确的政策利好、消息驱动或行业周期拐点
        - **板块效应**: 所在板块当天至少 **3只以上** 涨停股, 形成群体效应
        - **核心地位**: 必须是板块内的核心股 — 启动时最先涨停、封单最大、带动效应最强
        - **卡位风险**: 单打独斗的个股很难走远。若同题材内有竞争股存在, 需警惕卡位风险

        > 本脚本通过板块涨停家数统计 + 封板时间排名自动识别核心股
        """)

    with tab2:
        st.markdown("""
        ### 第二步: 量能配合 — 前缩后放, 前放后缩

        这是二进三最关键的判断维度:

        | 前两板形态 | 三板要求 | 逻辑 |
        |---|---|---|
        | **缩量涨停** (市场惜售) | **放量**, 消化获利抛压 | 获利盘未充分释放, 三板需换手化解潜在抛压 |
        | **放量涨停** (市场分歧大) | **缩量** (换手率<8%), 进入锁仓阶段 | 分歧转一致, 筹码锁定良好, 进入加速阶段 |

        > **二板放量是关键信号** — 市场分歧有助于题材走得更远
        """)

    with tab3:
        st.markdown("""
        ### 第三步: 位置与形态

        | 维度 | 具体要求 |
        |---|---|
        | 股价与市值 | **30元以下**、**市值200亿以内** 的中小盘股 |
        | 均线形态 | 5日、10日、20日均线**多头排列**, 股价在均线上方 |
        | 避免高位 | 近期有过**4连板以上**横盘的个股, 抛压沉重, 尽量避开 |
        | 强度递增 | 三板开盘高度/封板速度需强于二板 (如高开7%+早盘快速涨停) |
        """)

    with tab4:
        st.markdown("""
        ### 第四步: 排除杂音与龙虎榜审查

        - **过滤** ST、北交所、科创板个股
        - **查看前一日龙虎榜**: 前五买入席位合计占比 **不超过当日成交额的20%**, 否则容易成为次日砸盘力量
        - **关注前两板主力净额**: 前两板主力买入过多的个股, 接力资金容易被砸
        """)


# ========================================================================
#                       主入口
# ========================================================================

def run_yijiner(date_str: str):
    """执行一进二分析流程"""
    # 获取涨停池
    with st.spinner(f"正在获取 {date_str} 涨停股池数据..."):
        zt_df = fetch_zt_pool(date_str)

    if zt_df.empty:
        st.error(f"未获取到 {date_str} 的涨停股池数据, 请确认是否为交易日")
        return

    st.success(f"当日共有 {len(zt_df)} 只个股涨停")

    # 筛选首板
    col_lianban = get_col(zt_df, ["连板数"])
    if col_lianban:
        zt_first = zt_df[zt_df[col_lianban].fillna(1).astype(float) <= 1].copy()
    else:
        zt_first = zt_df.copy()
    st.info(f"其中 **首板** 个股: {len(zt_first)} 只")

    if zt_first.empty:
        st.warning("当日无首板个股")
        render_sector_analysis(zt_df)
        return

    # 补充振幅/量比 (涨停池本身不含这两个字段)
    spot_df = fetch_spot_market()
    if spot_df is not None:
        col_code_spot = get_col(spot_df, ["代码"])
        col_amp_spot = get_col(spot_df, ["振幅"])
        col_vr_spot = get_col(spot_df, ["量比"])
        keep_cols = [c for c in [col_code_spot, col_amp_spot, col_vr_spot] if c]
        if col_code_spot:
            spot_slim = spot_df[keep_cols].rename(
                columns={col_code_spot: "代码", col_amp_spot: "振幅", col_vr_spot: "量比"}
            )
            zt_first = zt_first.merge(spot_slim, on="代码", how="left")

    # 第一步: 减法排除
    st.divider()
    st.subheader("第一步: 减法排除")

    kept_df, dropped_df = pre_filter_yijiner(zt_first)

    col1, col2 = st.columns(2)
    col1.metric("通过初筛", len(kept_df))
    col2.metric("已排除", len(dropped_df))

    if not dropped_df.empty:
        with st.expander(f"查看已排除的 {len(dropped_df)} 只个股"):
            drop_cols = ["代码", "名称", "_drop_reasons"]
            available = [c for c in drop_cols if c in dropped_df.columns]
            st.dataframe(dropped_df[available], use_container_width=True, hide_index=True)

    if kept_df.empty:
        st.warning("没有通过初筛的标的")
        render_sector_analysis(zt_df)
        return

    # 第二步: 五维评分
    st.divider()
    st.subheader("第二步: 五维评分")

    col_code = get_col(kept_df, ["代码"])
    col_name = get_col(kept_df, ["名称"])

    results = []
    sector_cache = {}
    progress_bar = st.progress(0, text="正在分析...")
    total = len(kept_df)

    for i, (_, row) in enumerate(kept_df.iterrows()):
        code = str(row.get(col_code, "")).zfill(6)
        name = row.get(col_name, "N/A")

        progress_bar.progress(
            (i + 1) / total,
            text=f"分析中 ({i+1}/{total}): {code} {name}"
        )

        result = analyze_single_yijiner(code, name, row, zt_first, sector_cache)
        if result:
            results.append(result)

        time.sleep(0.15)

    progress_bar.empty()

    if not results:
        st.warning("分析未产生有效结果")
        return

    # 渲染结果
    st.divider()
    st.subheader("📊 一进二种子选手排名 (满分100)")

    render_results_table_yijiner(results, Config.TOP_N)

    # 个股详情
    st.divider()
    render_stock_detail_yijiner(results)

    # 板块分析
    st.divider()
    render_sector_analysis(zt_df)

    # 方法论
    st.divider()
    with st.expander("📖 一进二方法论说明"):
        render_methodology_yijiner()


def run_erjin3(date_str: str):
    """执行二进三分析流程"""
    with st.spinner(f"正在获取 {date_str} 涨停数据..."):
        zt_df = fetch_zt_pool(date_str)

    if zt_df.empty:
        st.error(f"未获取到 {date_str} 的涨停股池数据, 请确认是否为交易日")
        return

    st.success(f"当日共有 {len(zt_df)} 只个股涨停")

    col_lianban = get_col(zt_df, ["连板数"])
    if col_lianban:
        erjin3_df = zt_df[zt_df[col_lianban].fillna(0).astype(float) == 2].copy()
        st.info(f"其中 **2连板** 个股: {len(erjin3_df)} 只")
    else:
        st.warning("未找到'连板数'列, 无法筛选2连板个股")
        return

    if erjin3_df.empty:
        st.warning("当日无2连板个股")
        render_sector_analysis(zt_df)
        return

    # 第一步: 排除
    st.divider()
    st.subheader("第一步: 排除与过滤")

    kept_df, dropped_df = step1_erjin3_exclude_and_theme(erjin3_df)

    col1, col2 = st.columns(2)
    col1.metric("通过初筛", len(kept_df))
    col2.metric("已排除", len(dropped_df))

    if not dropped_df.empty:
        with st.expander(f"查看已排除的 {len(dropped_df)} 只个股"):
            drop_cols = ["代码", "名称", "_drop_reasons"]
            available = [c for c in drop_cols if c in dropped_df.columns]
            st.dataframe(dropped_df[available], use_container_width=True, hide_index=True)

    if kept_df.empty:
        st.warning("没有通过初筛的标的")
        render_sector_analysis(zt_df)
        return

    # 第二步~第四步: 逐股分析
    st.divider()
    st.subheader("第二步~第四步: 四维深度分析")

    lhb_df = fetch_lhb_data(date_str)

    col_code = get_col(kept_df, ["代码"])
    col_name = get_col(kept_df, ["名称"])

    results = []
    progress_bar = st.progress(0, text="正在分析...")
    total = len(kept_df)

    for i, (_, row) in enumerate(kept_df.iterrows()):
        code = str(row.get(col_code, "")).zfill(6)
        name = row.get(col_name, "N/A")

        progress_bar.progress(
            (i + 1) / total,
            text=f"分析中 ({i+1}/{total}): {code} {name}"
        )

        result = analyze_single_erjin3(code, name, row, zt_df, lhb_df)
        if result:
            results.append(result)

        time.sleep(0.15)

    progress_bar.empty()

    if not results:
        st.warning("分析未产生有效结果")
        return

    # 渲染结果
    st.divider()
    st.subheader("📊 二进三候选排名")

    render_results_table_erjin3(results, Config.TOP_N)

    # 个股详情
    st.divider()
    render_stock_detail_erjin3(results)

    # 板块分析
    st.divider()
    render_sector_analysis(zt_df)

    # 方法论
    st.divider()
    with st.expander("📖 二进三方法论说明"):
        render_methodology_erjin3()


def main():
    st.title("📈 A股涨停板筛选器")
    st.caption("一进二 (首板→二板) + 二进三 (二板→三板) | 数据来源: 东方财富 (AKShare)")

    # 侧边栏
    mode, date_str, config_overrides = render_sidebar()
    apply_config_overrides(config_overrides)

    is_yijiner = "一进二" in mode

    # 主区域
    btn_label = "🚀 开始分析 (一进二)" if is_yijiner else "🚀 开始分析 (二进三)"
    if not st.sidebar.button(btn_label, type="primary", use_container_width=True):
        # 首次加载, 显示方法论文档
        if is_yijiner:
            render_methodology_yijiner()
        else:
            render_methodology_erjin3()
        st.info("👈 在左侧选择日期和参数, 然后点击 **开始分析**")
        return

    # 根据模式分支
    if is_yijiner:
        run_yijiner(date_str)
    else:
        run_erjin3(date_str)

    # 页脚
    st.divider()
    st.caption(f"数据来源: 东方财富网 (AKShare) | 分析日期: {date_str} | 仅供研究参考, 不构成投资建议")


if __name__ == "__main__":
    # 如果直接用 python 运行, 自动转为 streamlit run 启动
    import os as _os, subprocess as _sp, sys as _sys, traceback as _tb
    if "STREAMLIT_RUNTIME" not in _os.environ:
        _sys.exit(_sp.call(["streamlit", "run", _os.path.abspath(__file__)] + _sys.argv[1:]))
    try:
        main()
    except Exception:
        st.error(f"应用启动异常:\n```\n{_tb.format_exc()}\n```")