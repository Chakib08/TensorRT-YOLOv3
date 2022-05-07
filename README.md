# Object Detection With The ONNX TensorRT Backend In Python using YOLOv3 and YOLOv3-Tiny 

**Table Of Contents**
- [Enviroments](#enviroments)
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Enviroments:

This repository was tested with the following environments 

1. NVIDIA Jetson AGX Xavier 
- [x] Python3
- [x] Jetpack 4.5.1
- [x] Ubuntu 18.04
- [x] CUDA 10.2
- [x] CuDNN 8.0
- [x] TensorRT 7.1.3
- [x] YOLOv3 and YOLOv3-Tiny

2. MSI GF65 Thin 10UE-284FR - NVIDIA GEFORCE RTX 3060
- [x] Python3
- [x] Ubuntu 20.04
- [x] CUDA 11.6
- [x] CuDNN 8.3.3
- [x] TensorRT 8.4.0
- [x] YOLOv3 and YOLOv3-Tiny

## Description

This is an edited NVIDIA sample about how to implement YOLOv3 and YOLOv3-Tiny using TensorRT to do benchmarks, the original can be found in the path **/usr/src/tensorrt/samples/python/yolov3_onnx/onnx_to_tensorrt.py**, so before executing this code, we have to execute [yolov3_to_onnnx.py] to parse the DarkNet model into ONNX model or import directly the onnx model from ONNX github, after the generation of the serialized model.onnx, we can run this code and specify the parameters like the model, resolution. For example to run a YOLOv3 model on the image kite.jpg with a 416x416 resolution and INT8 precision mode and a batch size of 1 we have to use this command :

`$ python3 onnx_to_tensorrt.py --image kite --model yolov3 --resolution 416 --precision INT8 --batch 1 --verbose`

CLI arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT, --image INPUT
                        Set the name of the input image
  -m MODEL, --model MODEL
                        Set the name of the model you want to use, <<yolov3>>
                        to use YOLOv3 or <<yolov3-tiny>> to use YOLOv3-Tiny
  -r RESOLUTION, --resolution RESOLUTION
                        Set the resolution of the input [608, 416 or 288]
  -p PRECISION, --precision PRECISION
                        Set the precision mode [FP32, FP16 or INT8]
  -b BATCH, --batch BATCH
                        Set The size of the batch
  -v, --verbose         Enable verbose to check the logs
.

## How does this sample work?

First, the original YOLOv3 specification from the paper is converted to the Open Neural Network Exchange (ONNX) format in `yolov3_to_onnx.py` (only has to be done once).

Second, this ONNX representation of YOLOv3 is used to build a TensorRT engine, followed by inference on a sample image in `onnx_to_tensorrt.py`. The predicted bounding boxes are finally drawn to the original input image and saved to disk.

After inference, post-processing including bounding-box clustering is applied. The resulting bounding boxes are eventually drawn to a new image file and stored on disk for inspection.

**Note:** This sample is not supported on Ubuntu 14.04 and older.

## Prerequisites

For specific software versions, see the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html).

1.  If you are using an NVIDIA Jetson board with the jetpack installed you can directly go to step 4 and run the `sh get_requirements.sh`
2.  If you are using a laptop with an NVIDIA GPU you have to install the following packages :

- [CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)

You can download and install CUDA according to the version of TensorRT you want to use, in my case i've installed CUDA 11.6 with the following commands :
	```
	$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
	$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
	$ wget https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb
	$ sudo dpkg -i cuda-repo-ubuntu2004-11-6-local_11.6.2-510.47.03-1_amd64.deb
	$ sudo apt-key add /var/cuda-repo-ubuntu2004-11-6-local/7fa2af80.pub
	$ sudo apt-get update
	$ sudo apt-get -y install cuda
	```
- [CuDNN](https://developer.nvidia.com/rdp/cudnn-archive)

You can download CuDNN in the link above and run the CLI below, in my case i've installed CuDNN 8.3.3 for CUDA 11.5, but it also worked for CUDA 11.6
	`sudo dpkg -i cudnn-local-repo-ubuntu2004-8.3.3.40_1.0-1_amd64.deb`
	
- [CUDA Toolkit]
	`sudo apt install nvidia-cuda-toolkit`

- [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download)

You can download the version of TensorRT you want to use, in my case i've donwloaded the 8.4.0 `nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.0.6-ea-20220212_1-1_amd64.deb` and i installed this later with commands below or you can just follow the [NVIDIA TensorRT instllation page](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). Make sure you have installed CUDA before installing TensorRT.
	```
	os="ubuntu2004"
	tag="cuda11.6-trt8.4.0.6-ea-20220212"
	sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
	sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub

	sudo apt-get update
	sudo apt-get install tensorrt`
	```
- [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us#) (Skip this part if your driver is already installed)

You can install the NVIDIA driver from the link above by selecting the reference of your GPU or installed it with the following CLI
	`sudo apt install nvidia-driver-510 nvidia-dkms-510`
	
3. This step is mandatory after step 2 to verify if all the needed packages was successfully installed

Run `nvidia-smi` on your terminal to check if the driver is correctly installed, you should get table below with the version of your driver and CUDA

	```
	+-----------------------------------------------------------------------------+
	| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
	|-------------------------------+----------------------+----------------------+
	| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
	| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
	|                               |                      |               MIG M. |
	|===============================+======================+======================|
	|   0  NVIDIA GeForce ...  On   | 00000000:01:00.0 Off |                  N/A |
	| N/A   39C    P8    10W /  N/A |    338MiB /  6144MiB |      0%      Default |
	|                               |                      |                  N/A |
	+-------------------------------+----------------------+----------------------+
	+-----------------------------------------------------------------------------+
	| Processes:                                                                  |
	|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
	|        ID   ID                                                   Usage      |
	|=============================================================================|
	|    0   N/A  N/A      1070      G   /usr/lib/xorg/Xorg                 45MiB |
	|    0   N/A  N/A      1636      G   /usr/lib/xorg/Xorg                161MiB |
	|    0   N/A  N/A      1807      G   /usr/bin/gnome-shell               28MiB |
	|    0   N/A  N/A      2779      G   ...566448415806487972,131072       91MiB |
	+-----------------------------------------------------------------------------+
	```
Run `nvcc -V` on your terminal to check CUDA Toolkit version

	```
	nvcc: NVIDIA (R) Cuda compiler driver
	Copyright (c) 2005-2019 NVIDIA Corporation
	Built on Sun_Jul_28_19:07:16_PDT_2019
	Cuda compilation tools, release 10.1, V10.1.243
	```
	
Run `dpkg -l | grep libnvinfer` to check the version of TensorRT, you should have the output below according to the version of TensorRT you have installed

	```
	ii  libnvinfer-bin                                              8.4.0-1+cuda11.6                    amd64        TensorRT binaries
	ii  libnvinfer-dev                                              8.4.0-1+cuda11.6                    amd64        TensorRT development libraries and headers
	ii  libnvinfer-doc                                              8.4.0-1+cuda11.6                    all          TensorRT documentation
	ii  libnvinfer-plugin-dev                                       8.4.0-1+cuda11.6                    amd64        TensorRT plugin libraries
	ii  libnvinfer-plugin8                                          8.4.0-1+cuda11.6                    amd64        TensorRT plugin libraries
	ii  libnvinfer-samples                                          8.4.0-1+cuda11.6                    all          TensorRT samples
	ii  libnvinfer8                                                 8.4.0-1+cuda11.6                    amd64        TensorRT runtime libraries
	ii  python3-libnvinfer                                          8.4.0-1+cuda11.6                    amd64        Python 3 bindings for TensorRT
	ii  python3-libnvinfer-dev                                      8.4.0-1+cuda11.6                    amd64        Python 3 development package for TensorRT
	```
	
Run `dpkg -l | grep cuda` to check the version of CUDA

Run `dpkg -l | grep cudnn` to check the version of CuDNN

4.  Install the dependencies for Python3 and download darknet weights and onnx models

	`$ sh get_requirements.sh`

## Running the sample

1.  Run on your terminal `sh get_requirements.sh` to get ONNX models if you haven't already run this command.

2.  Build a TensorRT engine from the generated ONNX file and run inference on a sample image and show with OpenCV the detection according of the index of the batch.
	For TensorRT v7 version use
	`python3 onnx_to_tensorrt_v7.py -i dog -m yolov3 -r 416 -p FP32 -b 1 --verbose`
	
	For TensorRT v8 version use
	`python3 onnx_to_tensorrt_v8.py -i dog -m yolov3 -r 416 -p FP32 -b 1 --verbose`

	Building an engine from file yolov3.onnx, this may take a while...
	Running inference on image dog.jpg...
	Saved image with bounding boxes of detected objects to dog_bboxes.jpg.
	```
	Reading engine from file tensorrt_engine/yolov3-tiny-416-INT8-bs16.trt
	Running inference on image test_images/dog.jpg...
	Latency = 15.57ms | FPS = 1027.73
	[[108.28476079 188.27050212 286.63875325 355.7010887 ]
	 [429.43957671  76.93088856 286.18962092  93.58978557]
	 [502.82192616  62.90276415 149.57946582 124.59640347]] [0.81173893 0.79401759 0.73235517] [16  2  2]
	Saved image with bounding boxes of detected objects to dog_out_yolov3-tiny_416_INT8_bs1.png.
	```
3. A window wil be opened which contain the output image with the bounding boxes of the detection according the whole parameters set with the performances and the image is saved in the yolov3_outputs folder or the yolov3_tiny_outputs depending of the model used in the CLI. Notice that you can run the model with FP16 or INT8 precision mode, change the batch and input resolution of the model.


# Additional resources

The following resources provide a deeper understanding about the model used in this sample, as well as the dataset it was trained on:

**Model**
- [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [YOLOv3-Tiny](https://pjreddie.com/darknet/yolo/)

**Dataset**
- [COCO dataset](http://cocodataset.org/#home)

**Documentation**
- [YOLOv3-608 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [Introduction To NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working With TensorRT Using The Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

September 2021
This `README.md` file was edited according to the one that is located in `/usr/src/tensorrt/samples/python/yolov3_onnx`.
# Known issues

If you use the CLI `nvidia-smi` and then you have the following output

`NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.`

You need to fix your Secure boot loader by rebooting your PC, and then go to the BIOS in the security section and you have to disable the security boot, check this [video](https://youtu.be/epCN8bSkYRg) to fix this issue.


	
	
	
	



