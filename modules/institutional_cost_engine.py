def add_cost(df):
    df = df.copy()
    vwap            = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["VWAP"]      = vwap
    df["CostBreak"] = (df["Close"] > df["VWAP"]).astype(int)
    return df
