
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README

---
library_name: segmentation-models-pytorch
license: other
pipeline_tag: image-classification
tags:
- segmentation-models-pytorch
- image-classification
- pytorch
- resnet
languages:
- python
---

# Model card for resnet50.

This repository contains the `imagenet` pre-trained weights for the `resnet50` model used as 
encoder in the [segmentation-models-pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) library.

### Example usage:

1. Install the library:

```bash
pip install segmentation-models-pytorch
```

2. Use the encoder in your code:

```python
import segmentation_models_pytorch as smp

model = smp.Unet("resnet50", encoder_weights="imagenet")
```

### References

- Github: https://github.com/qubvel/segmentation_models.pytorch
- Docs: https://smp.readthedocs.io/en/latest/
- Original weights URL: https://download.pytorch.org/models/resnet50-19c8e357.pth


Output:
{
    "extracted_code": "import segmentation_models_pytorch as smp\n\nmodel = smp.Unet(\"resnet50\", encoder_weights=\"imagenet\")"
}
