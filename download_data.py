from datasets import load_dataset
import time

def load_dataset_with_retries(dataset_name, config_name=None, max_retries=3, sleep_time=5):
    """Attempts to load a dataset with specified retries and sleep intervals.

    Args:
        dataset_name (str): The name of the dataset repository on Hugging Face.
        config_name (str, optional): The name of the dataset configuration.
        max_retries (int, optional): Maximum number of retry attempts.
        sleep_time (int, optional): Time to sleep between retries in seconds.

    Returns:
        Dataset: The loaded dataset object on success, None on failure.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to load the dataset
            ds = load_dataset(dataset_name, name=config_name)
            print(f"Dataset loaded successfully on attempt {attempt + 1}")
            return ds
        except Exception as e:
            print(f"Failed to load dataset on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Failed to load dataset.")
    return None

# Example usage
dataset_name = "togethercomputer/RedPajama-Data-V2"
config_name = "sample"
ds = load_dataset_with_retries(dataset_name, config_name=config_name, max_retries=50, sleep_time=10)

if ds is not None:
    # Dataset loaded successfully, proceed with your code
    pass
else:
    # Handle the failure to load the dataset
    pass

