import pandas as pd
from datasets import Dataset
from datasets import load_dataset as hg_load_dataset
from src.data import *


def extract_translation(dataset: object, lang_keys: list = ["en", "hi_en"]) -> dict:
    """
    Description: Extract the translation from the dataset.
    Assumption: The dataset must contain a column named "translation".
                Each row of the column "translation" must be a dictionary with keys as language codes and values as translations.
    Input parameters:
        - dataset: A dataset object.
    Returns: A dictionary containing the pandas dataframe dataset.
    """

    df = {}

    def process(df_split: pd.DataFrame, split: str = "train") -> None:
        df_split.loc[:, "en"] = df_split["translation"].apply(lambda x: x[lang_keys[0]])
        df_split.loc[:, "hi_en"] = df_split["translation"].apply(lambda x: x[lang_keys[1]])
        df_split = df_split[["en", "hi_en"]]
        df_split.columns = PROCESSED_COLUMN_NAMES
        df[split] = df_split

    if isinstance(dataset, Dataset):
        df_split = pd.DataFrame(dataset)
        process(df_split)
    else:
        for split in dataset.keys():
            df_split = pd.DataFrame(dataset[split])
            process(df_split, split=split)
    return df

def get_huggingface_dataset(dataset_name: str, split: str = None, lang_keys: list = ["en", "hi_en"]) -> dict:
    """
    Description: Loads a dataset from HuggingFace datasets library.
    Input parameters:
        - dataset_name: Name of the dataset to be loaded.
        - split: Name of the split to be loaded.
    Returns: A dictionary containing the dataset.
    """
    return extract_translation(hg_load_dataset(dataset_name, split=split), lang_keys=lang_keys)

def load_huggingface_dataset() -> dict:
    """
    Description: Loads a set of dataset from HuggingFace datasets library.
    Input parameters: None
    Returns: A dictionary containing the datasets.
    """
    df = {}
    for dataset_name in HUGGINGFACE_DATASET:
        if dataset_name == "cmu_hinglish_dog":
            df[dataset_name] = get_huggingface_dataset(dataset_name=dataset_name, lang_keys=["en", "hi_en"])
        elif dataset_name == "findnitai/english-to-hinglish":
            df[dataset_name] = get_huggingface_dataset(dataset_name=dataset_name, lang_keys=["en", "hi_ng"])
    return df

def get_hinglish_top_dataset(file_name: str, cols: list = ["en_query", "cs_query"]) -> pd.DataFrame:
    """
    Description: Loads a pandas df from Hinglish Top dataset, given a .tsv file.
    Input parameters:
        - file_name: Name of the file to be loaded.
        - cols: List of columns that must be considered when dropping rows and to be kept.
    Returns: A pandas df containing the dataset.
    """
    df = pd.read_table(os.path.join(BASE_DATA_DIR, file_name), skip_blank_lines=True, on_bad_lines='skip')[cols]
    df.dropna(subset=cols, inplace=True)
    df.columns = PROCESSED_COLUMN_NAMES
    return df

def load_hinglish_top_dataset(splits: list = ["train", "validation", "test"]) -> dict:
    """
    Description: Loads a set of pandas df from Hinglish Top dataset.
    Input parameters:
        - splits: List of splits to be loaded.
    Returns: A dictionary containing the datasets.
    """
    dataset = {}
    for split in splits:
        if split in HINGLISH_TOP_DATASET.keys():
            df = pd.DataFrame()
            for f in HINGLISH_TOP_DATASET[split]:
                df = pd.concat([df, get_hinglish_top_dataset(file_name=f)], axis=0)
            dataset[split] = df
    return {HINGLISH_TOP_DATA["name"]: dataset}

def get_linc_dataset(file_name: str) -> pd.DataFrame:
    """
    Description: Loads a pandas df from Linc dataset, given a .txt file.
    Input parameters:
        - file_name: Name of the file to be loaded.
    Returns: A pandas df containing the dataset.
    """
    with open(os.path.join(BASE_DATA_DIR, file_name), "r") as f:
        df = pd.DataFrame([line.split("\t") for line in f.readlines()])
    df.columns = PROCESSED_COLUMN_NAMES
    return df

def load_linc_dataset(splits: list = ["train", "validation", "test"]) -> dict:
    """
    Description: Loads a set of pandas df from Linc dataset.
    Input parameters:
        - splits: List of splits to be loaded.
    Returns: A dictionary containing the datasets.
    """
    dataset = {}
    for split in splits:
        if split in LINC_DATASET.keys():
            df = pd.DataFrame()
            for file_name in LINC_DATASET[split]:
                df = pd.concat([df, get_linc_dataset(file_name=file_name)], axis=0)
            dataset[split] = df
    return {LINC_DATA["name"]: dataset}

def load_dataset() -> dict:
    """
    Description: Load all the datasets mentioned in config.
                The resultant dictionary will be of the following format:
                {
                    dataset_name_1: {
                        train: train pandas df,
                        validation: validation pandas df,
                        test: test pandas df
                    },
                    dataset_name_2: {
                        train: train pandas df,
                        validation: validation pandas df,
                        test: test pandas df
                    }
                    ...
                }
    Returns: A dictionary containing the datasets.
    """
    return {**load_huggingface_dataset(), **load_hinglish_top_dataset(), **load_linc_dataset()}

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Description: Remove any duplicate rows from a dataframe
    Input parameters:
        - df: The input dataframe from which duplicates are to be removed.
    Returns: A dataframe containing no duplicate rows
    """
    duplicates = df.duplicated()
    has_duplicates = duplicates.any()
    if has_duplicates:
        df = df.drop_duplicates()
    return df

def get_dataset(remove: bool = True) -> tuple:
    """
    Description: Get all the datasets mentioned in config and save them. 
    Input parameters:
        - remove: If True, remove all the datasets from the processed data directory and reprocess them.
    Returns: A tuple containing train/validation/split pandas data frames.
    """
    data = load_dataset()
    train_df = validation_df = test_df = pd.DataFrame()
    for dataset in data.keys():
        if "train" in data[dataset].keys():
            train_df = pd.concat([train_df, data[dataset]["train"]], axis=0)
        if "validation" in data[dataset].keys():
            validation_df = pd.concat([validation_df, data[dataset]["validation"]], axis=0)
        if "test" in data[dataset].keys():
            test_df = pd.concat([test_df, data[dataset]["test"]], axis=0)
    train_df = clean_df(train_df)
    validation_df = clean_df(validation_df)
    test_df = clean_df(test_df)
    if os.path.exists(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TRAIN_CSV)) and remove:
        os.remove(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TRAIN_CSV))
    if not os.path.exists(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TRAIN_CSV)):
        train_df.to_csv(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TRAIN_CSV), index=False)
    else:
        train_df = pd.read_csv(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TRAIN_CSV))
    if os.path.exists(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_VALIDATION_CSV)) and remove:
        os.remove(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_VALIDATION_CSV))
    if not os.path.exists(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_VALIDATION_CSV)):
        validation_df.to_csv(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_VALIDATION_CSV), index=False)
    else:
        validation_df = pd.read_csv(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_VALIDATION_CSV))
    if os.path.exists(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TEST_CSV)) and remove:
        os.remove(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TEST_CSV))
    if not os.path.exists(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TEST_CSV)):
        test_df.to_csv(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TEST_CSV), index=False)
    else:
        test_df = pd.read_csv(os.path.join(PROCESSED_DATA_BASE_DIR, PROCESSED_TEST_CSV))
    return (train_df, validation_df, test_df)