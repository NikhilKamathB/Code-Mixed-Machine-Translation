import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CodeMixedDataset(Dataset):

    '''
        Custom dataset class for Code Mixed data
    '''

    def __init__(self, data: pd.DataFrame, overfit: bool = False, overfit_size: int = 32):
        '''
            Initial definition of the dataset
            Input params:
                data: pandas dataframe containing the data
                overfit: bool, if True, the dataset will be overfitted on a small sample
                overfit_size: int, size of the overfit dataset
        '''
        super().__init__()
        self.data = data
        self.overfit = overfit
        self.overfit_size = overfit_size

    def __len__(self):
        '''
            Returns the length of the dataset
        '''
        if self.overfit:
            return self.overfit_size
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        '''
            Returns a dict instance of the dataset
        '''
        raw_instance = self.data.iloc[index]
        instance = {
            "text": raw_instance["text"],
            "label": raw_instance["text"]
        }
        return instance


class CodeMixedDataLoader(DataLoader):

    '''
        Custom dataloader class for Code Mixed data
    '''

    def __init__(self,
                 train_df: pd.DataFrame = None,
                 val_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None,
                 train_batch_size: int = 32,
                 val_batch_size: int = 32,
                 test_batch_size: int = 32,
                 train_shuffle: bool = True,
                 val_shuffle: bool = False,
                 test_shuffle: bool = False,
                 overfit: bool = False, 
                 overfit_batch_size: int = 32,
                 num_workers: int = None,
                 pin_memory: bool = False,
                ) -> None:
        '''
            Initial definition of the dataloader
            Input params:
                train_df: pandas dataframe containing the training data
                val_df: pandas dataframe containing the validation data
                test_df: pandas dataframe containing the test data
                train_batch_size: int, batch size for training
                val_batch_size: int, batch size for validation
                test_batch_size: int, batch size for testing
                train_shuffle: bool, if True, the training data will be shuffled
                val_shuffle: bool, if True, the validation data will be shuffled
                test_shuffle: bool, if True, the test data will be shuffled
                overfit: bool, if True, the dataset will be overfitted on a small sample
                overfit_batch_size: int, batch size for overfitting
                num_workers: int, number of workers for the dataloader
                pin_memory: bool, if True, the data will be pinned to memory
        '''
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.overfit = overfit
        self.overfit_batch_size = overfit_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def get_dataloader(self) -> tuple:
        '''
            Returns a dataloader for the given dataset
            Input params: None
            Output params: a tuple of dataloaders for train, val and test
        '''
        train_dataset = CodeMixedDataset(
            data=self.train_df,
            overfit=self.overfit,
            overfit_size=self.overfit_batch_size
        )
        val_dataset = CodeMixedDataset(
            data=self.val_df,
            overfit=self.overfit,
            overfit_size=self.overfit_batch_size
        )
        test_dataset = CodeMixedDataset(
            data=self.test_df,
            overfit=self.overfit,
            overfit_size=self.overfit_batch_size
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=self.test_batch_size,
            shuffle=self.test_shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        return (train_dataloader, val_dataloader, test_dataloader)
    
    def visualize(self) -> None:
        '''
            This function is used to visualize the dataset results.
            Input params: None
            Output params: None
        '''
        train_dataloader, val_dataloader, test_dataloader = self.get_dataloader()
        print("Train Dataloader")
        print("Batch Size: ", self.train_batch_size)
        print("Number of batches: ", len(train_dataloader))
        print("Val Dataloader")
        print("Batch Size: ", self.val_batch_size)
        print("Number of batches: ", len(val_dataloader))
        print("Test Dataloader")
        print("Batch Size: ", self.test_batch_size)
        print("Number of batches: ", len(test_dataloader))