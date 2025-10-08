
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
language: en
license: mit
library_name: timm
tags:
- image-classification
- resnet18
- cifar100
datasets: cifar100
metrics:
- accuracy
model-index:
- name: resnet18_cifar100
  results:
  - task:
      type: image-classification
    dataset:
      name: CIFAR-100
      type: cifar100
    metrics:
    - type: accuracy
      value: 0.7926
---

# Model Card for Model ID

This model is a small resnet18 trained on cifar100.

- **Test Accuracy:** 0.7926
- **License:** MIT

## How to Get Started with the Model

Use the code below to get started with the model.

```python
import detectors
import timm

model = timm.create_model("resnet18_cifar100", pretrained=True)
```

## Training Data

Training data is cifar100.

## Training Hyperparameters


- **config**: `scripts/train_configs/cifar100.json`

- **model**: `resnet18_cifar100`

- **dataset**: `cifar100`

- **batch_size**: `128`

- **epochs**: `300`

- **validation_frequency**: `5`

- **seed**: `1`

- **criterion**: `CrossEntropyLoss`

- **criterion_kwargs**: `{}`

- **optimizer**: `SGD`

- **lr**: `0.1`

- **optimizer_kwargs**: `{'momentum': 0.9, 'weight_decay': 0.0005}`

- **scheduler**: `CosineAnnealingLR`

- **scheduler_kwargs**: `{'T_max': 280}`

- **debug**: `False`


## Testing Data

Testing data is cifar100.

---

This model card was created by Eduardo Dadalto.
Output:
{
    "extracted_code": "import detectors\nimport timm\n\nmodel = timm.create_model(\"resnet18_cifar100\", pretrained=True)\n"
}
