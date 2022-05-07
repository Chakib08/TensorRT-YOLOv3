"""onnx_to_tensorrt.py
The original code could be found in python TensorRT sample code:
"/usr/src/tensorrt/samples/python/yolov3_onnx/onnx_to_tensorrt.py".  I made the
modification so that we can do the inference and benchmark of convolutional neural networks like
YOLOv3 and YOLOv3-Tiny with different precision mode and different batch size.
"""

#!/usr/bin/env python2
#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

from __future__ import print_function

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import ImageDraw
from random import choice


from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

import sys, os

#sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

#print(sys.modules['common'])

from calibrator import YOLOEntropyCalibrator
import time
import argparse
import cv2


desc = ('This is an edited NVIDIA sample about how to implement YOLOv3 and YOLOv3-Tiny using TensorRT'
            ', before executing this code, we have to execute yolov3_to_onnnx.py to parse the DarkNet model into ONNX model'
            ', after the generation of the serialized model.onnx, we can run this code and specify the parameters like the model, resolution...'
        'For example to run a YOLOv3 model on the image dog.jpg with a 416x416 resolution and FP16 precision mode and a batch=1 we have to use this command : '
        '=========================================================================== python3 onnx_to_tensorrt.py -i dog -m yolov3 -r 416 -p FP16 -b 1')
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('-i', '--input', '--image', help="Set the name of the input image", type=str)
parser.add_argument('-m', '--model', help="Set the name of the model you want to use, <<yolov3>> to use YOLOv3 or <<yolov3-tiny>> to use YOLOv3-Tiny", type=str)
parser.add_argument('-r', '--resolution', help="Set the resolution of the input [608, 416 or 288]", type=str)
parser.add_argument('-p', '--precision', help="Set the precision mode [FP32, FP16 or INT8]", type=str)
parser.add_argument('-b', '--batch', help="Set The size of the batch", type=int)
parser.add_argument('-f', '--frames', help="Set number of frame for inference", type=int)

args = parser.parse_args()

# Set batch size
batch_size = 1
if args.batch is not None:
	batch_size = args.batch

# Set number of frame
nbr_frame = 1000
if args.frames is not None:
	nbr_frame = args.frames

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if args.verbose else trt.Logger()
colors = ['red','blue','green','yellow']

def draw_bboxes(image_raw, bboxes, confidences, categories, all_categories, bbox_color=choice(colors)):
    """Draw the bounding boxes on the original input image and return it.

    Keyword arguments:
    image_raw -- a raw PIL Image
    bboxes -- NumPy array containing the bounding box coordinates of N objects, with shape (N,4).
    categories -- NumPy array containing the corresponding category for each object,
    with shape (N,)
    confidences -- NumPy array containing the corresponding confidence for each object,
    with shape (N,)
    all_categories -- a list of all categories in the correct ordered (required for looking up
    the category name)
    bbox_color -- an optional string specifying the color of the bounding boxes (default: 'blue')
    """
    draw = ImageDraw.Draw(image_raw)
    print(bboxes, confidences, categories)
    for box, score, category in zip(bboxes, confidences, categories):
        x_coord, y_coord, width, height = box
        left = max(0, np.floor(x_coord + 0.5).astype(int))
        top = max(0, np.floor(y_coord + 0.5).astype(int))
        right = min(image_raw.width, np.floor(x_coord + width + 0.5).astype(int))
        bottom = min(image_raw.height, np.floor(y_coord + height + 0.5).astype(int))

        draw.rectangle(((left, top), (right, bottom)), outline=bbox_color)
        draw.text((left, top - 12), '{0} {1:.2f}'.format(all_categories[category], score), fill=bbox_color)

    return image_raw


def inference_param(model, resolution, precision, batch):
    """Take as parameter what the user set in the CLI, the inference will be executed according to these parameters

    :param model: The neural networks used for the inference (YOLOv3, YOLOv3-Tiny...)
    :param resolution: The input resolution of the model
    :param precision: TensorRT supported precision (FP32, FP16 or INT8)
    :param batch: The number of images in the input which will be processed at once
    """
    # Verify if the precision mode set by user is supported by TensorRT
    precisions = ["FP32","FP16","INT8"]
    for i in range(len(precisions)):
        if precision == precisions[i]:
            # Name the engine according to the parameters set in the CLI
            path_trt = model + "-" + resolution + "-" + precision + "-bs" + str(batch) + ".trt"
            # Selecting the input/outputs shapes and the ONNX file to parse
            if resolution == "608":
                input_resolution = (608, 608)
                input_shape = [batch, 3, 608, 608]

                if model == "yolov3":
                    output_shape = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)]
                    path_onnx = "yolov3-608.onnx"
                    calib_cache ='calibration_cache/calib_yolov3-int8-608.bin'

                elif model == "yolov3-tiny":
                    output_shape = [(1, 255, 19, 19), (1, 255, 38, 38)]
                    path_onnx = "yolov3-tiny-608.onnx"
                    calib_cache = 'calibration_cache/calib_yolov3-tiny-int8-608.bin'

            elif resolution == "416":
                input_resolution = (416, 416)
                input_shape = [batch, 3, 416, 416]

                if model == "yolov3":
                    output_shape = [(1, 255, 13, 13), (1, 255, 26, 26), (1, 255, 52, 52)]
                    path_onnx = "yolov3-416.onnx"
                    calib_cache = 'calibration_cache/calib_yolov3-int8-416.bin'

                elif model == "yolov3-tiny":
                    output_shape = [(1, 255, 13, 13), (1, 255, 26, 26)]
                    path_onnx = "yolov3-tiny-416.onnx"
                    calib_cache = 'calibration_cache/calib_yolov3-tiny-int8-416.bin'

            elif resolution == "288":
                input_resolution = (288, 288)
                input_shape = [batch, 3, 288, 288]

                if model == "yolov3":
                    output_shape = [(1, 255, 9, 9), (1, 255, 18, 18), (1, 255, 36, 36)]
                    path_onnx = "yolov3-288.onnx"
                    calib_cache = 'calibration_cache/calib_yolov3-int8-288.bin'

                elif model == "yolov3-tiny":
                    output_shape = [(1, 255, 9, 9), (1, 255, 18, 18)]
                    path_onnx = "yolov3-tiny-288.onnx"
                    calib_cache = 'calibration_cache/calib_yolov3-tiny-int8-288.bin'
        else:
            pass

    return input_resolution, input_shape, path_trt, path_onnx, output_shape, calib_cache


def get_engine(onnx_file_path, input_shape, input_res, calib_cache, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            trt.init_libnvinfer_plugins(None, "")

            builder.max_workspace_size = 2048  # 256MiB
            builder.max_batch_size = batch_size
            if args.precision == "FP16":
                builder.fp16_mode = True
                print("Using FP16 precision mode to build TensorRT engine...")
            if args.precision == "INT8":
                print("Using INT8 precision mode to build TensorRT engine...")
                builder.int8_mode = True
                builder.int8_calibrator = YOLOEntropyCalibrator('calib_images', input_res, calib_cache)
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = input_shape
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main():
    """Create a TensorRT engine for ONNX-based Model-Resolution-Precision Mode-Batch Size and run inference."""

    # Get the inference parameters which will be used to configure the TensorRT Builder
    try:
        input_res, input_shape, path_trt, path_onnx, output_shape, calib_cache = inference_param(args.model, args.resolution, args.precision,batch_size)
    except:
        print("""ERROR: Please verify that the parameters set in the CLI respect the arguments mentionned in the description and try again.
For more information use the commande : python3 onnx_to_tensorrt.py -h""")
        exit()

    # Load a previously generated network graph in ONNX format
    onnx_file_path = 'onnx_models/'+path_onnx
    engine_file_path = 'tensorrt_engine/'+path_trt
    # Load the image that we will be used to execute the inference
    input_img = args.input + ".jpg"
    input_image_path = 'test_images/'+input_img

    # Two-dimensional tuple with the target network's (spatial) input resolution in HW ordered
    input_resolution_yolov3_HW = input_res
    # Create a pre-processor object by specifying the required input resolution for YOLOv3
    preprocessor = PreprocessYOLO(input_resolution_yolov3_HW)
    # Load an image from the specified input path, and return it together with  a pre-processed version
    image_raw, image = preprocessor.process(input_image_path)
    # Store the shape of the original input image in WH format, we will need it for later
    shape_orig_WH = image_raw.size
    # Repeat the input image according to the batch size
    image = image.repeat(batch_size, axis=0)
    # Output shapes expected by the post-processor
    output_shapes = output_shape
    # Do inference with TensorRT
    trt_outputs = []
    with get_engine(onnx_file_path, input_shape, input_res, calib_cache, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        print('Running inference on image {}...'.format(input_image_path))
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inputs[0].host = image

        # Warmup
        trt_outputs_all = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs,
                                                 stream=stream)
        # Benchmark
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        stream.synchronize()
        nbr_frame = 1000
        start_time = time.time()
        for i in range(nbr_frame):
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
        latency = (time.time() - start_time) / nbr_frame
        fps = 1 / latency * batch_size
        print("Latency = {:.2f}ms | FPS = {:.2f}".format(latency * 1000, fps))

        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        trt_outputs_all = [out.host for out in outputs]

    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.

    for idx in range(batch_size):
        bs0 = np.prod(output_shapes[0])
        bs1 = np.prod(output_shapes[1])
        if args.model == "yolov3":
            bs2 = np.prod(output_shapes[2])
            trt_outputs = [trt_outputs_all[0][idx*bs0:(idx+1)*bs0], trt_outputs_all[1][idx*bs1:(idx+1)*bs1],trt_outputs_all[2][idx*bs2:(idx+1)*bs2]]
        elif args.model == "yolov3-tiny":
            trt_outputs = [trt_outputs_all[0][idx * bs0:(idx + 1) * bs0], trt_outputs_all[1][idx * bs1:(idx + 1) * bs1]]
        trt_outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]

        postprocessor_args = {"yolo_masks": [(6, 7, 8), (3, 4, 5), (0, 1, 2)],
                                  # A list of 3 three-dimensional tuples for the YOLO masks
                                  "yolo_anchors": [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                                   # A list of 9 two-dimensional tuples for the YOLO anchors
                                                   (59, 119), (116, 90), (156, 198), (373, 326)],
                                  "obj_threshold": 0.6,  # Threshold for object coverage, float value between 0 and 1
                                  "nms_threshold": 0.5,
                                  # Threshold for non-max suppression algorithm, float value between 0 and 1
                                  "yolo_input_resolution": input_resolution_yolov3_HW}

        postprocessor = PostprocessYOLO(**postprocessor_args)
        # Run the post-processing algorithms on the TensorRT outputs and get the bounding box details of detected objects
        boxes, classes, scores = postprocessor.process(trt_outputs, (shape_orig_WH))
        # Draw the bounding boxes onto the original input image and save it as a PNG file
        obj_detected_img = draw_bboxes(image_raw, boxes, scores, classes, ALL_CATEGORIES)

        output_image_path = args.input+"_out_"+args.model+"_"+args.resolution+"_"+args.precision+"_bs"+str(idx+1)+".png"
        if args.model == "yolov3":
            obj_detected_img.save('yolov3_outputs/' + output_image_path, 'PNG')
            img_boxes = 'yolov3_outputs/' + output_image_path

        elif args.model == "yolov3-tiny":
            obj_detected_img.save('yolov3_tiny_outputs/' + output_image_path, 'PNG')
            img_boxes = 'yolov3_tiny_outputs/' + output_image_path
        else:
            obj_detected_img.save(output_image_path, 'PNG')
            img_boxes = output_image_path
        print('Saved image with bounding boxes of detected objects to {}.'.format(output_image_path))
        # Display the bounding boxes of detected objects

        img_boxes = cv2.imread(img_boxes)
        #img_boxes = cv2.resize(img_boxes,(900,720))
        cv2.imshow("Model : {} | Precision : {} | Resolution : {}x{} | Batch size : {} | Batch index : {} | Latency : {:.2f}ms | FPS : {} ".
                   format(args.model, args.precision, args.resolution, args.resolution, args.batch, (idx + 1),latency * 1000, int(fps)), img_boxes)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()

