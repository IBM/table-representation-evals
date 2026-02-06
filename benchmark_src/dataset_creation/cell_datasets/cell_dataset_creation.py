
def create_cell_dataset(cfg, needs_download: bool = True):
    if cfg.dataset_name == "s2abel":
        from benchmark_src.dataset_creation.cell_datasets.s2abel_data import create_s2abel_dataset
        create_s2abel_dataset(cfg, needs_download)
    else:
        raise ValueError(f"Unknown cell dataset name: {cfg.dataset_name}")
    
    