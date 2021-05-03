#!/bin/bash

preprocess() {
        model_dir=$1
        arch=$2

        python hawq_utils_resnet.py --model-dir $model_dir --arch $arch
}

run_inference() {
	model_dir=$1
        arch=$2
        ###################Please set this path to your own imagenet validset##########
        val_dir="~/ILSVRC2012_img/val/"
	
	printf "%s\n" $model_dir

	python test_resnet_accuracy_imagenet.py --model-dir $model_dir --arch $arch --val-dir $val_dir > $model_dir/accuracy_result.txt 2>&1
}

preprocess "models/resnet18_uniform8"			"resnet18"
preprocess "models/resnet18_size0.75"			"resnet18"
preprocess "models/resnet18_size0.5"			"resnet18"
preprocess "models/resnet18_size0.25"			"resnet18"
preprocess "models/resnet18_bops0.75"			"resnet18"
preprocess "models/resnet18_bops0.5"			"resnet18"
preprocess "models/resnet18_bops0.25"			"resnet18"
preprocess "models/resnet18_latency0.75"		"resnet18"
preprocess "models/resnet18_latency0.5"			"resnet18"
preprocess "models/resnet18_latency0.25"		"resnet18"
preprocess "models/resnet18_uniform4"			"resnet18"

preprocess "models/resnet50_uniform8"			"resnet50"
preprocess "models/resnet50_size0.75"			"resnet50"
preprocess "models/resnet50_size0.5"			"resnet50"
preprocess "models/resnet50_size0.25"			"resnet50"
preprocess "models/resnet50_bops0.75"			"resnet50"
preprocess "models/resnet50_bops0.5"			"resnet50"
preprocess "models/resnet50_bops0.25"			"resnet50"
preprocess "models/resnet50_latency0.75"		"resnet50"
preprocess "models/resnet50_latency0.5"			"resnet50"
preprocess "models/resnet50_latency0.25"		"resnet50"
preprocess "models/resnet50_uniform4"			"resnet50"


run_inference "models/resnet18_uniform8"                "resnet18"
run_inference "models/resnet18_size0.75"		"resnet18"
run_inference "models/resnet18_size0.5"			"resnet18"
run_inference "models/resnet18_size0.25"		"resnet18"
run_inference "models/resnet18_bops0.75"		"resnet18"
run_inference "models/resnet18_bops0.5"			"resnet18"
run_inference "models/resnet18_bops0.25"		"resnet18"
run_inference "models/resnet18_latency0.75"		"resnet18"
run_inference "models/resnet18_latency0.5"		"resnet18"
run_inference "models/resnet18_latency0.25"		"resnet18"
run_inference "models/resnet18_uniform4"		"resnet18"

run_inference "models/resnet50_uniform8"                "resnet50"
run_inference "models/resnet50_size0.75"		"resnet50"
run_inference "models/resnet50_size0.5"			"resnet50"
run_inference "models/resnet50_size0.25"		"resnet50"
run_inference "models/resnet50_bops0.75"		"resnet50"
run_inference "models/resnet50_bops0.5"			"resnet50"
run_inference "models/resnet50_bops0.25"		"resnet50"
run_inference "models/resnet50_latency0.75"		"resnet50"
run_inference "models/resnet50_latency0.5"		"resnet50"
run_inference "models/resnet50_latency0.25"		"resnet50"
run_inference "models/resnet50_uniform4"		"resnet50"
