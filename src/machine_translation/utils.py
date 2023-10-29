import pandas as pd

def split_data(df: pd.DataFrame, split_ratio: list = [0.8, 0.1], random_state: int = 42) -> tuple:
    '''
        Split data into train, validation and test sets
        Input params:
            df: pandas dataframe
            split_ratio: list of split ratios. Total items in this list can be max 2, one corresponding
                        to train split and the other to validation split. Test split would
                        be the remaining. Sum of items in `split_ratio` must be less than 1.0.
            random_state: int -> Random state to use for shuffling the dataframe.
        Returns: tuple of dataframes - (train, val) if length of `split_ratio` is 1 and (train, val, test)
                if the length of `split_ratio` is 2.
    '''
    assert len(split_ratio) <= 2, 'Length of `split_ratio` must be less than or equal to 2.'
    assert sum(split_ratio) < 1.0, 'Sum of items in `split_ratio` must be less than 1.0.'
    df = df.sample(frac=1, random_state=random_state)
    train_split_ratio = split_ratio[0]
    if len(split_ratio) == 2:
        val_split_ratio = split_ratio[1]
    else:
        val_split_ratio = None
    train_df = df.iloc[:int(len(df)*train_split_ratio)]
    if val_split_ratio is None:
        val_df = df.iloc[int(len(df)*train_split_ratio): ]
        return (train_df, val_df)
    val_df = df.iloc[int(len(df)*train_split_ratio): int(len(df)*(train_split_ratio + val_split_ratio))]
    test_df = df.iloc[int(len(df)*(train_split_ratio + val_split_ratio)): ]
    return (train_df, val_df, test_df)