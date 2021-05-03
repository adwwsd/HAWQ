import os
from pathlib import Path

network_filename_list = [
        "resnet50_baseline",
        "resnet50_uniform8",
        "resnet50_size0.75",
        "resnet50_size0.5",
        "resnet50_size0.25",
        "resnet50_bops0.75",
        "resnet50_bops0.5",
        "resnet50_bops0.25",
        "resnet50_latency0.75",
        "resnet50_latency0.5",
        "resnet50_latency0.25",
        "resnet50_uniform4",
        "resnet18_baseline",
        "resnet18_uniform8",
        "resnet18_size0.75",
        "resnet18_size0.5",
        "resnet18_size0.25",
        "resnet18_bops0.75",
        "resnet18_bops0.5",
        "resnet18_bops0.25",
        "resnet18_latency0.75",
        "resnet18_latency0.5",
        "resnet18_latency0.25",
        "resnet18_uniform4"
        ]


link_dict = {
        "resnet18_baseline":"https://drive.google.com/file/d/1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE/view?usp=sharing",
        "resnet18_uniform8":"https://drive.google.com/file/d/1CLAd3LhiRVYwiBZRuUJgrzrrPFfLvfWG/view?usp=sharing",
        "resnet18_size0.75":"https://drive.google.com/file/d/1Fjm1Wruo773e3-jTIGahUWWQbmqGyMLO/view?usp=sharing",
        "resnet18_size0.5":"https://drive.google.com/file/d/1EGH76MRLckRtRXqWZHJ_I5DW5UQ-C8iA/view?usp=sharing",
        "resnet18_size0.25":"https://drive.google.com/file/d/1Eq9tmF8XlxOQGNMOuvc5rTPV0N4Ov-4C/view?usp=sharing",
        "resnet18_bops0.75":"https://drive.google.com/file/d/1F-pcK-AMCNcPAOydmhJN5aiDGGaEk-q7/view?usp=sharing",
        "resnet18_bops0.5":"https://drive.google.com/file/d/1DbDXYdulvvb9YOG1fRSrCVPvry_Reu8z/view?usp=sharing",
        "resnet18_bops0.25":"https://drive.google.com/file/d/1G9UgvLB3KuDyqNj4xV7DFiHjfXtULPJI/view?usp=sharing",
        "resnet18_latency0.75":"https://drive.google.com/file/d/1FcDVQT-p314lDq-URbHbLCSkGnWrd_vT/view?usp=sharing",
        "resnet18_latency0.5":"https://drive.google.com/file/d/1EfpPjgx-q5IS9rDP1irrdQtMvBodkDei/view?usp=sharing",
        "resnet18_latency0.25":"https://drive.google.com/file/d/1FwC7Sjp9lFW6dLdnyb9O4Re7OLkUpkPy/view?usp=sharing",
        "resnet18_uniform4":"https://drive.google.com/file/d/1D4DPcW2s9QmSnKzUgcjH-2eYO8zpDRIL/view?usp=sharing",
        "resnet50_baseline":"https://drive.google.com/file/d/1CE4b05gwMzDqcdpwHLFC2BM0841qKJp8/view?usp=sharing",
        "resnet50_uniform8":"https://drive.google.com/file/d/1Ldo51ZPx6_2Eq60JgbL6hdPdQf5WbRf9/view?usp=sharing",
        "resnet50_size0.75":"https://drive.google.com/file/d/1GtYgWFQrWfmn-23pFrZlxmBtuDCRG5Zs/view?usp=sharing",
        "resnet50_size0.5":"https://drive.google.com/file/d/1DnnRL9Q9SJ6BA5M98zGxcKrrAKClDdfJ/view?usp=sharing",
        "resnet50_bops0.75":"https://drive.google.com/file/d/1H5947bedQ1rCGzdKpSCJjIxysJUBznOE/view?usp=sharing",
        "resnet50_bops0.5":"https://drive.google.com/file/d/1DNUkyavD10saZw9_7TzJhEy0NFPhSVZr/view?usp=sharing",
        "resnet50_bops0.25":"https://drive.google.com/file/d/1G_JQJgGTDYQN5atmcyjDsJZV5zkH8GWw/view?usp=sharing",
        "resnet50_latency0.75":"https://drive.google.com/file/d/1HBQhrTplhOHft43WEifaq35dfUftP5tJ/view?usp=sharing",
        "resnet50_latency0.5":"https://drive.google.com/file/d/1GbviN74Z806jyDusohusEjgKuqIyAc5s/view?usp=sharing",
        "resnet50_latency0.25":"https://drive.google.com/file/d/1HuMaFhL1GV3XiYt9fLncZf6QruL7eGif/view?usp=sharing",
        "resnet50_uniform4":"https://drive.google.com/file/d/1DDis-8C-EupCRj-ExH58ldSv-tG2RXyf/view?usp=sharing"
        }



def filename_to_quantization_scheme(filename):
    return filename.split('_')[1].replace('0','_0').replace('size','modelsize')
def filename_to_arch(filename):
    return filename.split('_')[0]


for network_filename in network_filename_list:
    Path(f"models").mkdir(parents=True, exist_ok=True)
    if "baseline" in network_filename:
        continue
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


