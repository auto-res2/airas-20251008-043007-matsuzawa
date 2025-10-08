
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: label
    dtype:
      class_label:
        names:
          '0': airplane
          '1': automobile
          '2': bird
          '3': cat
          '4': deer
          '5': dog
          '6': frog
          '7': horse
          '8': ship
          '9': truck
  splits:
  - name: brightness
    num_bytes: 113527976
    num_examples: 50000
  - name: contrast
    num_bytes: 88391274
    num_examples: 50000
  - name: defocus_blur
    num_bytes: 103073344
    num_examples: 50000
  - name: elastic_transform
    num_bytes: 107021331
    num_examples: 50000
  - name: fog
    num_bytes: 103503757
    num_examples: 50000
  - name: frost
    num_bytes: 118892727
    num_examples: 50000
  - name: gaussian_blur
    num_bytes: 100670122
    num_examples: 50000
  - name: gaussian_noise
    num_bytes: 142658988
    num_examples: 50000
  - name: glass_blur
    num_bytes: 119706292
    num_examples: 50000
  - name: impulse_noise
    num_bytes: 123708719
    num_examples: 50000
  - name: jpeg_compression
    num_bytes: 113466166
    num_examples: 50000
  - name: motion_blur
    num_bytes: 102408130
    num_examples: 50000
  - name: pixelate
    num_bytes: 89632017
    num_examples: 50000
  - name: saturate
    num_bytes: 112404473
    num_examples: 50000
  - name: shot_noise
    num_bytes: 127391654
    num_examples: 50000
  - name: snow
    num_bytes: 119467934
    num_examples: 50000
  - name: spatter
    num_bytes: 121035992
    num_examples: 50000
  - name: speckle_noise
    num_bytes: 138559718
    num_examples: 50000
  - name: zoom_blur
    num_bytes: 101616964
    num_examples: 50000
  download_size: 2156114754
  dataset_size: 2147137578
configs:
- config_name: default
  data_files:
  - split: brightness
    path: data/brightness-*
  - split: contrast
    path: data/contrast-*
  - split: defocus_blur
    path: data/defocus_blur-*
  - split: elastic_transform
    path: data/elastic_transform-*
  - split: fog
    path: data/fog-*
  - split: frost
    path: data/frost-*
  - split: gaussian_blur
    path: data/gaussian_blur-*
  - split: gaussian_noise
    path: data/gaussian_noise-*
  - split: glass_blur
    path: data/glass_blur-*
  - split: impulse_noise
    path: data/impulse_noise-*
  - split: jpeg_compression
    path: data/jpeg_compression-*
  - split: motion_blur
    path: data/motion_blur-*
  - split: pixelate
    path: data/pixelate-*
  - split: saturate
    path: data/saturate-*
  - split: shot_noise
    path: data/shot_noise-*
  - split: snow
    path: data/snow-*
  - split: spatter
    path: data/spatter-*
  - split: speckle_noise
    path: data/speckle_noise-*
  - split: zoom_blur
    path: data/zoom_blur-*
---

Output:
{
    "extracted_code": ""
}
