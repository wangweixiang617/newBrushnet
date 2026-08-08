import baostock as bs
import pandas as pd
from datetime import datetime

# -------------------------- 已更新为三元组配置 --------------------------
ETF_CONFIG = [
    # ("中证A500ETF", "sh.512050", "中证A500"),
    # ("创业板指ETF", "sz.159915", "创业板指"),
    # ("科创50ETF", "sh.588000", "科创50"),
    # ("恒生科技ETF", "sh.513180", "恒生科技"),
    # ("港股通科技ETF", "sz.159636", "港股通科技"),
    # ("纳指100ETF", "sh.513100", "纳指100"),
    # ("中证银行ETF", "sh.512800", "中证银行"),
    # ("红利低波ETF", "sh.512890", "红利低波"),
    # ("黄金ETF", "sh.518880", "黄金9999"),

    # ("中证500", "sh.000905"),
    # ("中证A500", "sz.000510"),
    # ("创业板指", "sz.399006"),
    ("科创50ETF", "sh.588000"),
    ("中证银行ETF", "sh.512800"),
    # ("红利低波ETF", "sh.512890"),
    ("黄金ETF", "sh.518880"),
    ("恒生科技ETF", "sh.513180",),
    ("港股通科技ETF", "sz.159636"),
    ("纳指100ETF", "sh.513100"),
    ("新能电池ETF", "sz.159755"),
    ("电网设备ETF", "sz.159326")
]

# 已更新全局起始日期
GLOBAL_START_DATE = "2021-01-05"
END_DATE = datetime.now().strftime("%Y-%m-%d")


# -------------------------- 技术指标计算函数 --------------------------
def calculate_macd(df, prefix=""):
    ema_fast = df["close"].ewm(span=10, adjust=False).mean()
    ema_slow = df["close"].ewm(span=22, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    df.loc[:, f"{prefix}dif"] = dif
    df.loc[:, f"{prefix}dea"] = dea
    df.loc[:, f"{prefix}macd"] = macd
    return df

def calculate_kdj(df, prefix=""):
    low_list = df["low"].rolling(9, min_periods=1).min()
    high_list = df["high"].rolling(9, min_periods=1).max()

    # 计算高低价差，做除零保护
    hl_range = high_list - low_list
    # 价差为0时替换为空值，避免零除报错
    hl_range = hl_range.replace(0, pd.NA)

    rsv = (df["close"] - low_list) / hl_range * 100
    # 异常值填充50（多空平衡中性值），再前向填充保证指标连续
    rsv = rsv.fillna(50).ffill()

    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d

    df.loc[:, f"{prefix}k"] = k
    df.loc[:, f"{prefix}d"] = d
    df.loc[:, f"{prefix}j"] = j
    return df


# -------------------------- 周线指标生成函数 --------------------------
def add_weekly_indicators(daily_df):
    df = daily_df.copy()

    # 强制确保date列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        print("  ⚠️  检测到date列不是datetime类型，正在重新转换...")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("所有日期转换失败，无法继续处理")

    df.loc[:, "date"] = df["date"].dt.normalize()

    weekly_df = df.resample("W-FRI", on="date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "amount": "sum"
    }).reset_index()

    weekly_df.loc[:, "date"] = weekly_df["date"].dt.normalize()
    weekly_df = calculate_macd(weekly_df, prefix="week_")
    weekly_df = calculate_kdj(weekly_df, prefix="week_")

    df.loc[:, "week_end_date"] = df["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()

    weekly_merge = weekly_df[["date", "week_dif", "week_dea", "week_macd", "week_k", "week_d", "week_j"]].copy()
    weekly_merge.columns = ["week_end_date", "week_dif", "week_dea", "week_macd", "week_k", "week_d", "week_j"]
    weekly_merge.loc[:, "week_end_date"] = weekly_merge["week_end_date"].dt.normalize()

    df = pd.merge(df, weekly_merge, on="week_end_date", how="left")
    df = df.drop(columns=["week_end_date"])

    # 前向填充周线指标
    df[["week_dif", "week_dea", "week_macd", "week_k", "week_d", "week_j"]] = df[
        ["week_dif", "week_dea", "week_macd", "week_k", "week_d", "week_j"]
    ].ffill()

    matched_count = df["week_dif"].notna().sum()
    print(f"  周线指标匹配成功：{matched_count}/{len(df)} 行")

    return df


# -------------------------- 金叉死叉状态计算 --------------------------
def calculate_cross_status(df):
    print("  正在计算金叉死叉状态...")

    # 日线MACD状态
    df["day_macd_cross"] = 0
    df.loc[
        (df["day_dif"] > df["day_dea"]) & (df["day_dif"].shift(1) <= df["day_dea"].shift(1)),
        "day_macd_cross"
    ] = 1
    df.loc[
        (df["day_dif"] < df["day_dea"]) & (df["day_dif"].shift(1) >= df["day_dea"].shift(1)),
        "day_macd_cross"
    ] = -1
    df.loc[0, "day_macd_cross"] = 1 if df["day_dif"].iloc[0] > df["day_dea"].iloc[0] else -1
    df["day_macd_status"] = df["day_macd_cross"].replace(0, pd.NA).ffill().astype(int)

    # 日线KDJ状态
    df["day_kdj_cross"] = 0
    df.loc[
        (df["day_k"] > df["day_d"]) & (df["day_k"].shift(1) <= df["day_d"].shift(1)),
        "day_kdj_cross"
    ] = 1
    df.loc[
        (df["day_k"] < df["day_d"]) & (df["day_k"].shift(1) >= df["day_d"].shift(1)),
        "day_kdj_cross"
    ] = -1
    df.loc[0, "day_kdj_cross"] = 1 if df["day_k"].iloc[0] > df["day_d"].iloc[0] else -1
    df["day_kdj_status"] = df["day_kdj_cross"].replace(0, pd.NA).ffill().astype(int)

    # 周线MACD状态
    df["week_macd_cross"] = 0
    df.loc[
        (df["week_dif"] > df["week_dea"]) & (df["week_dif"].shift(1) <= df["week_dea"].shift(1)),
        "week_macd_cross"
    ] = 1
    df.loc[
        (df["week_dif"] < df["week_dea"]) & (df["week_dif"].shift(1) >= df["week_dea"].shift(1)),
        "week_macd_cross"
    ] = -1
    df.loc[0, "week_macd_cross"] = 1 if df["week_dif"].iloc[0] > df["week_dea"].iloc[0] else -1
    df["week_macd_status"] = df["week_macd_cross"].replace(0, pd.NA).ffill().astype(int)

    # 周线KDJ状态
    df["week_kdj_cross"] = 0
    df.loc[
        (df["week_k"] > df["week_d"]) & (df["week_k"].shift(1) <= df["week_d"].shift(1)),
        "week_kdj_cross"
    ] = 1
    df.loc[
        (df["week_k"] < df["week_d"]) & (df["week_k"].shift(1) >= df["week_d"].shift(1)),
        "week_kdj_cross"
    ] = -1
    df.loc[0, "week_kdj_cross"] = 1 if df["week_k"].iloc[0] > df["week_d"].iloc[0] else -1
    df["week_kdj_status"] = df["week_kdj_cross"].replace(0, pd.NA).ffill().astype(int)

    # 删除临时列
    df = df.drop(columns=["day_macd_cross", "day_kdj_cross", "week_macd_cross", "week_kdj_cross"])

    return df


# -------------------------- 数据获取主函数 --------------------------
def get_etf_full_data(code, name):
    print(f"\n{'=' * 60}")
    print(f"正在处理：{name}")
    print(f"ETF代码：{code}")
    print('-' * 60)

    # 每个ETF单独登录登出
    login_result = bs.login()
    if login_result.error_code != '0':
        print(f"❌ Baostock登录失败：{login_result.error_msg}")
        return None

    try:
        # 获取上市日期
        start_date = GLOBAL_START_DATE
        basic_info = bs.query_stock_basic(code=code)
        if basic_info.error_code == '0':
            while basic_info.next():
                row_data = basic_info.get_row_data()
                ipo_date_raw = row_data[2]
                if len(ipo_date_raw) == 8 and ipo_date_raw.isdigit():
                    ipo_date = f"{ipo_date_raw[:4]}-{ipo_date_raw[4:6]}-{ipo_date_raw[6:8]}"
                    start_date = max(GLOBAL_START_DATE, ipo_date)
                    print(f"ℹ️  上市日期：{ipo_date}，实际起始日期：{start_date}")

        print(f"📅  请求数据时间范围：{start_date} 至 {END_DATE}")

        # 获取行情数据
        rs = bs.query_history_k_data_plus(
            code=code,
            fields="date,open,high,low,close,volume,amount,pctChg,turn",
            start_date=start_date,
            end_date=END_DATE,
            frequency="d",
            adjustflag="2"
        )

        if rs.error_code != '0':
            print(f"❌ 获取行情数据失败：{rs.error_msg}")
            return None

        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            print(f"❌ 未获取到任何行情数据")
            return None

        df = pd.DataFrame(data_list, columns=rs.fields)

        # 强制日期转换并验证
        print("  正在转换日期格式...")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).reset_index(drop=True)

        if df.empty:
            print(f"❌ 所有数据日期转换失败")
            return None

        print(f"✅ 日期转换成功，有效数据：{len(df)} 行")

        # 数据类型转换
        for col in df.columns:
            if col != "date":
                df.loc[:, col] = pd.to_numeric(df[col], errors="coerce")

        print(f"✅ 原始数据获取成功：共 {len(df)} 个交易日")
        print(f"数据起始日期：{df['date'].iloc[0].strftime('%Y-%m-%d')}")
        print(f"数据结束日期：{df['date'].iloc[-1].strftime('%Y-%m-%d')}")

        # 计算技术指标
        print("正在计算日线MACD/KDJ...")
        df = calculate_macd(df, prefix="day_")
        df = calculate_kdj(df, prefix="day_")

        print("正在计算周线MACD/KDJ并映射到日线...")
        df = add_weekly_indicators(df)

        # 计算金叉死叉状态
        df = calculate_cross_status(df)

        print(f"\n🎉 {name} 处理完成！")
        print(f"总数据条数：{len(df)}")
        rate = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        print(f"累计涨跌幅：{rate:.2f}%")

        # 打印最新状态
        latest = df.iloc[-1]
        print(f"\n最新状态（{latest['date'].strftime('%Y-%m-%d')}）：")
        print(f"  日线MACD：{'金叉' if latest['day_macd_status'] == 1 else '死叉'}")
        print(f"  日线KDJ：{'金叉' if latest['day_kdj_status'] == 1 else '死叉'}")
        print(f"  周线MACD：{'金叉' if latest['week_macd_status'] == 1 else '死叉'}")
        print(f"  周线KDJ：{'金叉' if latest['week_kdj_status'] == 1 else '死叉'}")

        return df

    except Exception as e:
        print(f"❌ 处理失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # 确保登出
        try:
            bs.logout()
        except:
            pass


# -------------------------- 已更新主程序循环 --------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Baostock ETF 完整数据工具（配置更新版）")
    print(f"全局起始日期：{GLOBAL_START_DATE}")
    print(f"结束日期：{END_DATE}")
    print("=" * 60)

    all_data = {}
    success_count = 0

    # 已修改为三元组循环
    for name, code in ETF_CONFIG:
        df = get_etf_full_data(code, name)
        if df is not None and not df.empty:
            all_data[name] = df
            success_count += 1

    # 导出所有数据
    print("\n" + "=" * 60)
    print(f"全部处理完成：成功{success_count}/{len(ETF_CONFIG)}个ETF")
    print("=" * 60)

    if all_data:
        # 打印数据统计
        print("\n📊 所有ETF真实数据统计：")
        print("-" * 80)
        print(f"{'指数名称':<18}{'起始日期':<14}{'结束日期':<14}{'数据条数':<10}{'最新日MACD':<12}{'最新周MACD':<12}")
        print("-" * 80)

        for name, df in all_data.items():
            start = df['date'].iloc[0].strftime('%Y-%m-%d')
            end = df['date'].iloc[-1].strftime('%Y-%m-%d')
            count = len(df)
            dm = "金叉" if df['day_macd_status'].iloc[-1] == 1 else "死叉"
            wm = "金叉" if df['week_macd_status'].iloc[-1] == 1 else "死叉"
            print(f"{name:<18}{start:<14}{end:<14}{count:<10}{dm:<12}{wm:<12}")

        print("-" * 80)

        # 导出Excel（使用配置中的第三个字段作为Sheet名称）
        filename = f"Baostock_8大指数_ETF数据_含金叉死叉_{datetime.now().strftime('%Y%m%d')}.xlsx"
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            for name, df in all_data.items():
                df.to_excel(writer, sheet_name=name[:31], index=False)

        print(f"\n✅ 所有数据已导出到：{filename}")
        print("\nExcel Sheet名称与对应指数：")
        for name, code in ETF_CONFIG:
            print(f"  - {name}")
        print("\n每行最后4列状态：")
        print("  day_macd_status ：1=金叉，-1=死叉")
        print("  day_kdj_status  ：1=金叉，-1=死叉")
        print("  week_macd_status：1=金叉，-1=死叉")
        print("  week_kdj_status ：1=金叉，-1=死叉")
    else:
        print("\n⚠️  未获取到任何有效数据")