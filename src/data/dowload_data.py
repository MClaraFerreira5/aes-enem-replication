from datasets import load_dataset
import os

def download_aes_enem():
    output_dir = "data/raw/aes_enem"
    os.makedirs(output_dir, exist_ok=True)
    
    configs = ['sourceAWithGraders', 'sourceB']
    
    for config_name in configs:
        print(f"Baixando a config '{config_name}'...")
        ds = load_dataset('kamel-usp/aes_enem_dataset', config_name)
        
        for split in ds.keys():
            file_path = os.path.join(output_dir, f"{config_name}_{split}.csv")
            ds[split].to_pandas().to_csv(file_path, index=False)
            print(f"  Salvo: {file_path}")

if __name__ == "__main__":
    download_aes_enem()