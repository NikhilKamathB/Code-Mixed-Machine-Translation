import os
import shutil
from transformers import TrainerCallback
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState
from src.machine_translation.utils import upload_blob


class GCPCallback(TrainerCallback):

    '''
        Callback to handle GCP specific tasks. Tasks covered so far:
        1. Storing checkpoints to GCP bucket.
    '''

    def _clear_directory(self, directory: str) -> None:
        '''
            Clears the directory.
            Input params:
                directory: str -> The directory to clear.
        '''
        shutil.rmtree(directory)

    def __init__(self, save_to_gcp: bool = True, clear_local_storage: bool = True, destination_path: str = None, verbose: bool = True) -> None:
        '''
            Initializes the callback.
            Input params:
                save_to_gcp: bool -> Whether to save to GCP or not.
                clear_local_storage: bool -> Whether to clear local storage or not.
                destination_path: str -> The destination path to store the checkpoints.
                verbose: bool -> Whether to print the logs or not.
        '''
        self.save_to_gcp = save_to_gcp
        self.bucket_name = None
        self.parent_root = None
        self.clear_local_storage = clear_local_storage
        self.destination_path = destination_path
        self.verbose = verbose
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        '''
            Callback function to initialize the callback, gets called after the Trainer is initialized.
            Input params:
                args: TrainingArguments -> The training arguments.
                state: TrainerState -> The trainer state.
                control: TrainerControl -> The trainer control.
        '''
        self.bucket_name = os.getenv("GOOGLE_BUCKET_NAME", None)
        if self.bucket_name is None:
            raise ValueError("GOOGLE_BUCKET_NAME not found in environment variables.")
        self.parent_root = args.output_dir.split("/")[-1]

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        '''
            Callback function to save the checkpoint to GCP, gets called for every checkpoint save.
            Input params:
                args: TrainingArguments -> The training arguments.
                state: TrainerState -> The trainer state.
                control: TrainerControl -> The trainer control.
        '''
        if self.save_to_gcp:
            checkpoint = os.listdir(args.output_dir)[-1] # Get the latest checkpoint
            local_dir = os.path.join(args.output_dir, checkpoint)
            for local_file in os.listdir(local_dir):
                local_file_path = os.path.join(local_dir, local_file)
                remote_path = os.path.join(self.destination_path, self.parent_root, checkpoint, local_file)
                if self.verbose:
                    print(f"\nStep - {state.global_step}: Uploading {local_file_path} to {remote_path}...")
                _ = upload_blob(self.bucket_name, local_file_path, remote_path)
            if self.clear_local_storage: self._clear_directory(local_dir)