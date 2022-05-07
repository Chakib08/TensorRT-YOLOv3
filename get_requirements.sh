#!/bin/bash


sudo apt-get install python3
sudo apt-get install python3-pip

echo "Installing all python dependencies..."
pip3 install opencv-python
pip3 install numpy
pip3 install protobuf
pip3 install onnx
pip3 install pycuda
pip3 install pillow
pip3 install wget

echo "Installing megatools package..."
sudo apt install megatools

mkdir calib_images
echo "Downloading calibration caches"
wget "https://drive.google.com/uc?export=download&id=170uWN7AMSA6m-iUMglxs3oZpZQrJBPbb" -O calibration_cache.tar.gz
tar xzvf calibration_cache.tar.gz calibration_cache
rm calibration_cache.tar.gz

echo "Downloading darknet weights..."
megadl 'https://mega.nz/#!6Ho2mIgL!1ZxJZ_Ntm5imfNKZKH12CWNKRmstsUOZh2tAClx99hA'
tar xzvf darknet_weights.tar.gz darknet_weights
rm darknet_weights.tar.gz

echo "Downloading onnx models..."
megadl 'https://mega.nz/#!iOImFYhA!c8xBgOxDu2OT6AiCl0sjkRoTNNxLoJ4TyGMZ7f3jUoo'
tar xzvf onnx_models.tar.gz onnx_models
rm onnx_models.tar.gz

mkdir tensorrt_engine
mkdir yolov3_outputs
mkdir yolov3_tiny_outputs
