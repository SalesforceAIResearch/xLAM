## Data Conversion Guide

#### Activate Conda Environment

Ensure the actionstudio conda environment is active.

```bash
conda activate actionstudio
```

#### Convert Unified Datasets to Train-Ready Data

Run the following command to convert datasets:

```bash
python3 data_converters.py
```

#### Customize the Path for Generated Train Data

To specify a custom path for your train data, use:

```bash
python3 data_converters.py --train_data_path <YOUR_TRAIN_PATH>
```

#### Note for Data Quality 

**Note**: Low-quality trajectories of each dataset have been removed, ensuring the `unified_data` directory mainly contains primarily high-quality data.

### Convert Customized Data

- **Format Your Data**: Convert your data to the unified format. Refer to any dataset in the `xLAM/actionstudio/datasets/unified_data` directory for guidance.

- **Add Your Data**: Place your formatted data in the `xLAM/actionstudio/datasets/unified_data` folder.

- **Update Data Keys**: Add your dataset name as a key to the `use_data_keys` section in `data_converters.py`.