# Model definitions

The repository uses:

- **HARU-Net** as the high-capacity teacher (`HarUnet_model.py`, class `HARU_net`).
- **ResU-Net** as the lightweight student (`ResUnet_model.py`, class `ResUNet`).

The knowledge-distillation script imports the architectures as:

```python
from models.HarUnet_model import HARU_net
from models.ResUnet_model import ResUNet
```

The model class names are intentionally left unchanged from the experimental
implementations; only the filenames have been standardized for this repository.
