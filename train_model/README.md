# &#128204; Train on Google Colab

## &#9989; Our Environment
- anaconda3
- pytorch 1.1.0
- torchvision 0.3.0
- cuda 9.0
- cupy 6.0.0
- opencv-python 4.5.1
- 1 GPU
- python 3.6


## &#9989; Installation

### Install CONDA on Google Colab
```python
! wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh
! chmod +x Miniconda3-py37_4.9.2-Linux-x86_64.sh
! bash ./Miniconda3-py37_4.9.2-Linux-x86_64.sh -b -f -p /usr/local
import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')
```

### Make requirements.txt
```python
# # requirements.txt 생성
p = """
gensim==3.8.1
pyLDAvis==2.1.2
spacy==2.2.3
scikit-learn==0.23.1
seaborn==0.11.0
squarify==0.4.3
ipykernel
nltk
pandas
scipy
"""

c = """text_file = open("requirements.txt", "w+");text_file.write(p);text_file.close()""" 

exec(c)
```

### Make Venv
```python
!conda create -n tryon python=3.6
```

### Install requirements.txt
```python
!conda install --channel conda-forge --file requirements.txt --yes
```

### Install on Conda
```python
!conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
!conda install cupy or pip install cupy==6.0.0
```

### Make requirements2.txt
```python
# # requirements2.txt 생성
p = """
torch
Pillow
torchvision
tensorboardX
opencv-python
cupy-cuda101==6.1.0
"""

r = """text_file = open("requirements2.txt", "w+");text_file.write(p);text_file.close()""" 

exec(r)
```

### Install on Local
```python
!pip install -r requirements2.txt
```

## &#9989; Git clone
```python
!git clone https://github.com/geyuying/PF-AFN.git
```

## &#9989; Set train~.sh file
- We use only 1 gpu
- So, Open the train~.sh file and Edit the nums gpu
```python
--nproc_per_node=1, --num_gpus 1
```


## &#9989; Train
```python
! chmod 770 train_PBAFN_stage1.sh
! chmod 770 train_PBAFN_e2e.sh
! chmod 770 train_PFAFN_stage1.sh
! chmod 770 train_PFAFN_e2e.sh
```
```python
! ./scripts/train_PBAFN_stage1.sh
```
```python
! ./scripts/train_PBAFN_e2e.sh
```
```python
! ./scripts/train_PFAFN_stage1.sh
```
```python
! ./scripts/train_PFAFN_e2e.sh
```