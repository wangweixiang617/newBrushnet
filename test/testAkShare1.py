import akshare as ak
import pandas as pd

# 获取贵州茅台(600519)日线数据，前复权
df = ak.stock_zh_a_hist(
    symbol="600519",
    period="daily",  # 周期：daily=日线, weekly=周线, monthly=月线
    start_date="20160101",
    end_date="20260610",
    adjust="hfq"  # 复权方式：hfq=前复权, qfq=后复权, ""=不复权
)

# 重命名列（方便后续使用）
# df.columns = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
# df["date"] = pd.to_datetime(df["date"])

# 保存到Excel
df.to_excel("贵州茅台日线数据.xlsx", index=False)
print(df.head())