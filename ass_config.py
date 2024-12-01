from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 50,
        "lr": 10**-4,
        "seq_len": 30,
        "d_model": 32,
        "datasource": 'date_translation',
        "lang_src": "src",
        "lang_tgt": "tgt",
        "model_folder": "weights",
        "model_basename": "date_transformer_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "train_data_path": "/home/harikrishnam/dl_nlp/a2/Assignment2_train.txt",
        "validation_data_path": "/home/harikrishnam/dl_nlp/a2/Assignment2_validation.txt",
        "experiment_name": "runs/date_transformer"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
