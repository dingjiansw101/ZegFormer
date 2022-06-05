This directory contains few tools to convert ImageNet pre-trained weights.

* `convert-torchvision-to-d2.py`

Tool to convert torchvision pre-trained weights for D2.

```
wget https://download.pytorch.org/models/resnet101-63fe2227.pth
python tools/convert-torchvision-to-d2.py resnet101-63fe2227.pth R-101.pkl
```
