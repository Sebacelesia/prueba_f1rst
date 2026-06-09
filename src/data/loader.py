import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    return df.reset_index(drop=True)
