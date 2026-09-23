# Model definitions

The manuscript experiments use:

- **HARU-Net** as the high-capacity teacher.
- **ResU-Net** as the lightweight student.

The supplied notebooks import:

```python
from HARUnet_model_v2_1 import HARU_net
from ResUNet_model import ResUNet
```

Add the exact `HARUnet_model_v2_1.py` and `ResUNet_model.py` files used in the experiments here (or update the imports consistently). They were not included in the supplied source files used to assemble this package, so no substitute architecture has been invented.

The manuscript describes the ResU-Net as a four-stage residual encoder-decoder with skip connections, a 1024-channel bottleneck for 256 x 256 inputs, residual convolutional encoder/decoder blocks, strided-convolution downsampling, and transposed-convolution upsampling.
