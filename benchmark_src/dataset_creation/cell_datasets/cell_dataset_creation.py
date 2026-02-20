
def create_cell_dataset(cfg, needs_download: bool = True):
    if "s2abel" in cfg.dataset_name:
        from benchmark_src.dataset_creation.cell_datasets.s2abel_data import create_s2abel_dataset
        create_s2abel_dataset(cfg, needs_download)
    else:
        raise ValueError(f"Unknown cell dataset name: {cfg.dataset_name}")
    
    