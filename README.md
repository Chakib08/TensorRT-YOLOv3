# Object Detection With The ONNX TensorRT Backend In Python using YOLOv3 and YOLOv3-Tiny 

**Table Of Contents**
- [Enviroments](#Enviroments)
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)
- [folders and scripts] (#folders-and-scripts)

## Enviroments:

This repository was tested with the following environments 

1. NVIDIA Jetson AGX Xavier 
- [x] Python3
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

1.  Install the dependencies for Python3 and download darknet weights and onnx models

	`$ sh get_requirements.sh`

## Running the sample

1.  Create an ONNX version of YOLOv3 with the following command. The Python script will also download all necessary files from the official mirrors (only once).
	`python yolov3_to_onnx.py`

	When running the above command for the first time, the output should look similar to the following:
	```
	Downloading from https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg, this may take a while...
	100% [................................................................................] 8342 / 8342 
	Downloading from https://pjreddie.com/media/files/yolov3.weights, this may take a while...
	100% [................................................................................] 248007048 / 248007048
	[...]
	%106_convolutional = Conv[auto_pad = u'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]]
	(%105_convolutional_lrelu, %106_convolutional_conv_weights, %106_convolutional_conv_bias)
	return %082_convolutional, %094_convolutional,%106_convolutional
	}
	```

2.  Build a TensorRT engine from the generated ONNX file and run inference on a sample image and show with OpenCV the detection according of the index of the batch.
	`python onnx_to_tensorrt.py`

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
3. A window wil be opened which contain the output image with the bounding boxes of the detection according the whole parameters set with the performances and the image is saved in the yolov3_outputs folder or the yolov3_tiny_outputs depending of the model used in the CLI.



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

There are no known issues in this sample.


# Folders and scripts

**Scripts**
	[data_processing.py] -> Include the preprocess and the post-process (Non-maximum Suppression) for both of YOLOv3 et YOLOv3-Tiny.
	[common.py] -> Include functions which allow us to run inference of the model on an image using CUDA.
	[change_batch.py] -> Change the batch of the input tensor of an ONNX neural networks.
	[calibrator.py] -> Include a class named YOLOEntropyCalibrator that allows to calibrate an ONNX model with calibration-set of 1000 images for INT8 Quantization.
	[yolov3_to_onnx] -> Generate YOLOv3 with ONNX format, to get YOLOv3-Tiny you check on ONNX github or generate it with **https://github.com/jkjung-avt/tensorrt_demos/tree/master/yolo** github.
	[onnx_to_tensorrt] -> The main script of the project which get edited to do benchmarks with of YOLOv3 and YOLOv3-Tiny according to the parameters set in the CLI as resolution, precision and batch.

**Folders**
	[calib_images] -> Include the calibration-set of 1000 images for INT8 Quantization (the folder is empty due to memory of the target).
	[calibration_caches] -> The calibration caches generated for with the algorithm **IInt8EntropyCalibrator2** included in the TensorRT Python API **https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/EntropyCalibrator2.html**.
	[darknet_weights] -> Include the congiguration (.cfg) and the weights (.weights) files of DarkNet-53 YOLOv3.
	[onnx_models] -> Include models with ONNX format.
	[tensorrt_engine] -> Include the optimized models generated with TensorRT.
	[test_images] -> Some test images to run the inference.
	[yolov3_outputs] -> Output images with YOLOv3 TensorRT inference, each image has a name according with the parameters set for the inference : imageName_out_yolov3_resolution_precision_batchIndex.png
	[yolov3_tiny_outputs] -> Output images with YOLOv3-Tiny TensorRT inference, each image has a name according with the parameters set for the inference : imageName_out_yolov3-tiny_resolution_precision_batchIndex.png
	
	
	
	



