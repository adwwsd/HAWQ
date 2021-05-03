import os
from pathlib import Path

network_filename_list = [
        "resnet50_baseline",
        "resnet18_baseline",
        ]


link_dict = {
        "resnet18_baseline":"https://drive.google.com/file/d/1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE/view?usp=sharing",
        "resnet50_baseline":"https://drive.google.com/file/d/1CE4b05gwMzDqcdpwHLFC2BM0841qKJp8/view?usp=sharing",
        }


def filename_to_quantization_scheme(filename):
    return filename.split('_')[1].replace('0','_0').replace('size','modelsize')
def filename_to_arch(filename):
    return filename.split('_')[0]


for network_filename in network_filename_list:
    #Path(f"models/{network_filename}").mkdir(parents=True, exist_ok=True)
    fileid = link_dict[network_filename].split('/')[5]
    tar_filename = f"models/{network_filename}.tar.gz"
    os.system(f"wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={fileid}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={fileid}\" -O {tar_filename} && rm -rf /tmp/cookies.txt")
    os.system(f"tar -zxvf models/{network_filename}.tar.gz --directory models")
    os.system(f"rm models/{network_filename}.tar.gz")
    
    resnet_arch = filename_to_arch(network_filename)
    quantization_scheme = filename_to_quantization_scheme(network_filename)

    ######## Please set this path to your imagenet validset
    path_to_imagenet = "~/ILSVRC2012_img"

    os.system(f"export CUDA_VISIBLE_DEVICES=0")
    os.system(f"python ../quant_train.py -a {resnet_arch} --epochs 1 --lr 0.0001 --batch-size 128 --data {path_to_imagenet} --save-path models/{network_filename}/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 1 --checkpoint-iter -1 --quant-scheme {quantization_scheme} --resume models/{network_filename}/checkpoint.pth.tar --resume-quantize -e")

