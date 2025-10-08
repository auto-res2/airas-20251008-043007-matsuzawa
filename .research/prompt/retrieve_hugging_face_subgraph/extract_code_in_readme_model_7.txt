
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
tags:
- image-classification
- timm
- transformers
library_name: timm
license: apache-2.0
datasets:
- imagenet-1k
- imagenet-21k
---
# Model card for vit_base_patch16_224.augreg2_in21k_ft_in1k

A Vision Transformer (ViT) image classification model. Trained on ImageNet-21k by paper authors and (re) fine-tuned on ImageNet-1k with additional augmentation and regularization by Ross Wightman.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 86.6
  - GMACs: 16.9
  - Activations (M): 16.5
  - Image size: 224 x 224
- **Papers:**
  - How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers: https://arxiv.org/abs/2106.10270
  - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: https://arxiv.org/abs/2010.11929v2
- **Dataset:** ImageNet-1k
- **Pretrain Dataset:** ImageNet-21k
- **Original:** https://github.com/google-research/vision_transformer

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_base_patch16_224.augreg2_in21k_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 197, 768) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

## Citation
```bibtex
@article{steiner2021augreg,
  title={How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers},
  author={Steiner, Andreas and Kolesnikov, Alexander and and Zhai, Xiaohua and Wightman, Ross and Uszkoreit, Jakob and Beyer, Lucas},
  journal={arXiv preprint arXiv:2106.10270},
  year={2021}
}
```
```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
Output:
{
    "extracted_code": "from urllib.request import urlopen\nfrom PIL import Image\nimport timm\n\nimg = Image.open(urlopen(\n    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'\n))\n\nmodel = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)\nmodel = model.eval()\n\n# get model specific transforms (normalization, resize)\ndata_config = timm.data.resolve_model_data_config(model)\ntransforms = timm.data.create_transform(**data_config, is_training=False)\n\noutput = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1\n\ntop5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)\n\nfrom urllib.request import urlopen\nfrom PIL import Image\nimport timm\n\nimg = Image.open(urlopen(\n    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'\n))\n\nmodel = timm.create_model(\n    'vit_base_patch16_224.augreg2_in21k_ft_in1k',\n    pretrained=True,\n    num_classes=0,  # remove classifier nn.Linear\n)\nmodel = model.eval()\n\n# get model specific transforms (normalization, resize)\ndata_config = timm.data.resolve_model_data_config(model)\ntransforms = timm.data.create_transform(**data_config, is_training=False)\n\noutput = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor\n\n# or equivalently (without needing to set num_classes=0)\n\noutput = model.forward_features(transforms(img).unsqueeze(0))\n# output is unpooled, a (1, 197, 768) shaped tensor\n\noutput = model.forward_head(output, pre_logits=True)\n# output is a (1, num_features) shaped tensor"
}
