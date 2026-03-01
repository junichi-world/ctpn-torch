# text-detection-ctpn

text detection mainly based on ctpn (connectionist text proposal network). It is implemented in tensorflow. I use id card detect as an example to demonstrate the results, but it should be noticing that this model can be used in almost every horizontal scene text detection task. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/). If you got any questions, check the issue first, if the problem persists, open a new issue.
***
# roadmap
- [x] freeze the graph for convenient inference
- [x] pure python, cython nms and cuda nms
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
***
# demo
- for a quick demo, use the SavedModel-based `demo_pb.py` inference entrypoint.
- first, git clone git@github.com:eragonruan/text-detection-ctpn.git --depth=1
- ensure there is a checkpoint under `ctpn/checkpoints` (or set `TEST.checkpoints_path` in `ctpn/text.yml`)
- export SavedModel by running:
```shell
python ./ctpn/generate_pb.py
```
- put your images in `data/demo`, then run:
```shell
python ./ctpn/demo_pb.py
```
***
# portable setup (docker / other machine)
- this repository alone is not enough for training or full inference; you also need runtime dependencies and model/data files.
- included in this repo now:
  - `Dockerfile` (GPU-ready base image path can be overridden)
  - `scripts/setup_env.sh` (installs Python deps and builds `lib/utils` extensions)
  - `scripts/run_demo.sh`, `scripts/run_train.sh`
- not included (you must place them manually):
  - training dataset in VOC layout: `data/VOCdevkit2007/VOC2007/...`
  - VGG pretrained weights: `data/pretrain/VGG_imagenet.npy`
  - checkpoints for inference (either `ctpn/checkpoints/` or `output/...`)

## run with docker (gpu)
- build image:
```shell
docker build -t ctpn-tf-gpu .
```
- run container with GPU and mount your project:
```shell
docker run --rm -it --gpus all \
  -v "$(pwd)":/workspace/text-detection-ctpn \
  -w /workspace/text-detection-ctpn \
  ctpn-tf-gpu bash
```
- inside container, initialize once (rebuilds `lib/utils` against the current Python):
```shell
./scripts/setup_env.sh
```
- run demo:
```shell
./scripts/run_demo.sh
```
- run training:
```shell
./scripts/run_train.sh
```

## run without docker (wsl/linux)
- use Python 3.10/3.11 and install TensorFlow (GPU or CPU) for your environment.
- then run:
```shell
python -m pip install -r requirements.txt
python -m pip install --force-reinstall "numpy<2"
./scripts/setup_env.sh --skip-pip
```
- `numpy<2` is important for compatibility with many TensorFlow builds used by this project.

## switching demo checkpoint
- `ctpn/demo.py` reads `TEST.checkpoints_path` from `ctpn/text.yml` and loads the latest checkpoint in that directory.
- examples:
  - bundled checkpoint: `checkpoints/`
  - your trained model: `output/ctpn_end2end/voc_2007_trainval`
***
# parameters
there are some parameters you may need to modify according to your requirement, you can find them in ctpn/text.yml
- USE_GPU_NMS # whether to use nms implemented in cuda or not
- DETECT_MODE # H represents horizontal mode, O represents oriented mode, default is H
- checkpoints_path # the model I provided is in checkpoints/, if you train the model by yourself,it will be saved in output/
***
# training
## setup
- requirements: python3.10/3.11, tensorflow==2.16.1, cython, opencv-python, easydict
- if you do not have a gpu device,follow here to [setup](https://github.com/eragonruan/text-detection-ctpn/issues/43)
- if you have a gpu device, build the library by
```shell
cd lib/utils
chmod +x make.sh
./make.sh
```
## prepare data
- First, download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. you can download it from [google drive](https://drive.google.com/drive/folders/0B_WmJoEtfQhDRl82b1dJTjB2ZGc?resourcekey=0-OjW5DtLUbX5xUob7fwRvEw&usp=sharing) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). 
- Second, prepare the training data as referred in paper, or you can download the data I prepared from [google drive](https://drive.google.com/drive/folders/0B_WmJoEtfQhDRl82b1dJTjB2ZGc?resourcekey=0-OjW5DtLUbX5xUob7fwRvEw&usp=sharing) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). Or you can prepare your own data according to the following steps. 
- Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run
```shell
cd lib/prepare_training_data
python split_label.py
```
- it will generate the prepared data in current folder, and then run
```shell
python ToVoc.py
```
- to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/ and then run
```shell
cd ../../data
ln -s TEXTVOC VOCdevkit2007
```
## train 
Simplely run
```shell
python ./ctpn/train_net.py
```
- you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
- The model I provided in checkpoints is trained on GTX1070 for 50k iters.
- If you are using cuda nms, it takes about 0.2s per iter. So it will takes about 2.5 hours to finished 50k iterations.
***
# some results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/data/results/006.jpg" width=320 height=480 /><img src="/data/results/008.jpg" width=320 height=480 />
<img src="/data/results/009.jpg" width=320 height=480 /><img src="/data/results/010.png" width=320 height=320 />
***
## oriented text connector
- oriented text connector has been implemented, i's working, but still need futher improvement.
- left figure is the result for DETECT_MODE H, right figure for DETECT_MODE O
<img src="/data/results/007.jpg" width=320 height=240 /><img src="/data/oriented_results/007.jpg" width=320 height=240 />
<img src="/data/results/008.jpg" width=320 height=480 /><img src="/data/oriented_results/008.jpg" width=320 height=480 />
***
