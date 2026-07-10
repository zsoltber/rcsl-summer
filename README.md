# RCSL Internship - Evolutionary fine-tuning on the edge and mask potency evaluation

## Multi-bit Continual Fine-tuning Flow Process

### 1. Train quantised model using Brevitas, synthesize into bitstream using FINN

**Important note:** Make sure `auto_folding_config.json` file is modified after generating folding estimates, such that each `MatrixVectorActivation_X` field has `"runtime_writeable_weights": 1`. Recommended to save this config to a separate file if `auto_folding_config.json` is regenerated during the build phase.

### 2. Pull NumPy weights of base model and copy build directory onto development board

If intermediate models are available, the integer NumPy layers can be extracted from `<build_dir>/intermediate_models/step_streamline.onnx` using a code snippet akin to:

```python 
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper

MODEL_PATH = '<build_dir>/intermediate_models/step_streamline.onnx' # replace <build_dir> with own build directory
SAVE_DIR = '<save_directory>' # replace <build_dir> with own build directory

streamline_model = ModelWrapper(MODEL_PATH)
integer_layers = []

for i in model.graph.initializer:
    init_name = i.name
    if 'MatMul' in init_name:
        integer_layers.append(model.get_initializer(init_name))
```

The resulting `integer_layers` is a list of `np.array` objects. They can be saved as `.npy` files into the `runtime_weights` directory where the `.dat` weights are found for ease of access. Once the destination folder is copied, it is worthwhile to make a copy of the `runtime_weights` folder to distinguish base weights from the weights used to perform and evaluate tuning operations.

#### Additional files on the FPGA

Add the `.npy` to `.dat` weight layer packing util file `packing.py` and the Jupyter Notebook `evolutionary.ipynb` to the output directory on the board.

## Important Files

### `packing.py` or `npy_to_finn_dat.py`

Responsible for converting NumPy weight arrays to `.dat` file required by FINN.

`packing.py` > Fully NumPy based and thus can be used on 32-bit PS boards too such as the PYNQ Z2.

Main callable function is `pack_layer()` with parameters:
-  `npy_layer`: NumPy array either directly or as a path to a `.npy` file
- `out_path`: `.dat` output file path
- `pe`, `simd`: PE, SIMD folding integer values for given layer, preferably extracted from `final_hw_config.json`
- `wdt`: Weight data type as a string, e.g. `'INT4'`
- `mw`, `mh`: Input weight array height and width
- `mode`: `'decoupled_runtime'` or `'decoupled_verilog_dat'`

    

### Optim algorithms

<!-- > [!NOTE]
> what does this look like

> [!WARNING]
> what does this look like

> [!TIP]
> what does this look like

---

- [x] is this a checkbox
- [ ] is this a checkbox



<details>
<summary>Click to show more details</summary>

Additional information goes here.

</details>


`topk_gft.py`
`topk_gft_lowmem.py`
`dropout_exploration.py` -->
