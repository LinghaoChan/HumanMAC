# Naive Motion Retargeting From AMASS to Human3.6M

## 🌲 File Tree (Before retargeting)
```bash
./retargeting
├── README.md
├── amass.npy
├── h36-real.npy
├── trans.py
├── vis-amass.py
├── vis-h36.py
└── vis-retar-h36.py
```
The `*.npy` file should be downloaded from [Google Drive](https://drive.google.com/drive/folders/1IkG41Qt36w3-A6jwBnfbUxckB742A7mN?usp=sharing).

## 🔧 Perform Retargeting
Please run the code:
```bash
python trans.py
```
It will retarget AMSSS from `amass.npy` to `retargeted.npy`. 

## 🎬 Visualization
Please run the code:
```bash
python vis-{DATA}.py
```
The `DATA` should be `amass`, `h36`, and `retar-h36` for AMASS dataset, Human3.6M dataset, and retargeted dataset respectively. 

## 🌲 File Tree (After retargeting and visualization)
```bash
./retargeting
├── vis/*.png
├── README.md
├── amass.npy
├── h36-real.npy
├── retargeted.npy
├── trans.py
├── vis-amass.py
├── vis-h36.py
├── vis-retar-h36.py
└── vis.mp4
```