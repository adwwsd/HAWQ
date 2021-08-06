import os
import sys

import mxnet as mx
import numpy as np
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse

#import torch.cuda.profiler as profiler
#import pyprof
#pyprof.init()

parser = argparse.ArgumentParser(description='Mixed precision resnet example',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size')

parser.add_argument('--num-layers', type=int, default=50,
                    help='Number of layers in Resnet')

parser.add_argument('--data-layout', default='HWNC',
                    help='Data layout (NHWC, NCWH)')

parser.add_argument('--kernel-layout', default='HWOI',
                    help='Kernel layout (OIHW, HWOI)')

parser.add_argument('--with-bn', action='store_true', default=False,
                    help='Add batch normalization to the model')

args = parser.parse_args()

###############################################################################
# Prepare Resnet model
# -----------------


resnet18 = ptcv_get_model("resnet18", pretrained=True)
resnet50 = ptcv_get_model("resnet50", pretrained=True)

#resnet18_params = torch.load("models/resnet18_baseline/resnet18.pth")
#resnet50_params = torch.load("models/resnet50_baseline/resnet50.pth")

device = torch.device("cuda")
#resnet18.load_state_dict(resnet18_params)
#resnet50.load_state_dict(resnet50_params)
resnet18.to(device)
resnet50.to(device)

batch_size = args.batch_size

###############################################################################
# Inference time measure
# -----------------
### ResNet18 Infernce Time Check
repetitions=2000
dummy_input = torch.randn(batch_size,3,224,224,dtype=torch.float).to(device)
total_time=0
for _ in range(50):
    _ = resnet18(dummy_input)

with torch.no_grad():
    for rep in range(repetitions):
        starter,ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()
        _ = resnet18(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        total_time += curr_time
time_average = total_time/(repetitions*batch_size)

print(f"resnet18 time_average:{time_average* 1000}ms")

### ResNet50 Infernce Time Check
repetitions=2000
total_time=0
for _ in range(50):
    _ = resnet50(dummy_input)

iter_to_capture = 50
with torch.no_grad():
    #with torch.autograd.profiler.emit_nvtx():
    for rep in range(repetitions):
        starter,ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #if rep == iter_to_capture:
            #profiler.start();
        starter.record()
        _ = resnet50(dummy_input)
        ender.record()
        #if rep == iter_to_capture:
            #profiler.stop();
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)/1000
        total_time += curr_time

time_average = total_time/(repetitions*batch_size)

print(f"resnet50 time_average:{time_average* 1000}ms")


'''

###############################################################################
# Prepare validation data
# -----------------

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
batch_size=8
val_dir = "~/ILSVRC2012_img/val/"

val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
            ])),
        batch_size=batch_size, shuffle=False,num_workers=4,pin_memory=True)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
acc_top1.reset()
acc_top5.reset()

resnet18.eval()

with torch.no_grad():
    for i, (images,target) in enumerate(val_loader):
        images, target = images.to(device), target.to(device)
        output = resnet18(images)                   
        
        acc_top1.update([mx.nd.array(target.cpu().numpy())], [mx.nd.array(output.cpu().detach().numpy())])
        acc_top5.update([mx.nd.array(target.cpu().numpy())], [mx.nd.array(output.cpu().detach().numpy())])
        
        if not (i+1) % 10:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            print("resnet18 - [%d samples] validation: acc-top1=%f acc-top5=%f" % (nsamples, top1, top5))

                
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
acc_top1.reset()
acc_top5.reset()

resnet50.eval()

with torch.no_grad():
    for i, (images,target) in enumerate(val_loader):
        images, target = images.to(device), target.to(device)
        output = resnet50(images)                   
        
        acc_top1.update([mx.nd.array(target.cpu().numpy())], [mx.nd.array(output.cpu().detach().numpy())])
        acc_top5.update([mx.nd.array(target.cpu().numpy())], [mx.nd.array(output.cpu().detach().numpy())])
        
        if not (i+1) % 10:
            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            nsamples = (i + 1) * batch_size
            print("resnet50 - [%d samples] validation: acc-top1=%f acc-top5=%f" % (nsamples, top1, top5))
'''
