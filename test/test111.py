import baostock as bs
import pandas as pd

# 只登录一次
lg = bs.login()
print("login:", lg.error_code, lg.error_msg)

codes = [
    # ("中证A500", "sh.000905")
    # ("中证A500", "sz.000510")
    # ("创业板指", "sz.399006"),
    # ("科创50", "sh.000688")
    ("电池ETF", "sz.159755"),
    ("电网设备ETF", "sz.159326")
]

for name, code in codes:
    rs = bs.query_history_k_data_plus(
        code=code,
        fields="date,close",
        start_date="2021-06-12",
        end_date="2026-06-12",
        frequency="d"
    )
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    print(name, code, "rows:", len(data))

bs.logout()