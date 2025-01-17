#!/bin/bash


run_inference() {
	bit_config=$1
	num_layers=$2
    batch_size=$3
    data_layout=$4

	
	printf "========%s========\n" $bit_config

        # If you'd like only to measure inference time without autoTVM support, run:
	python test_resnet_inference_time.py --bit-config $bit_config --num-layers $num_layers --batch-size $batch_size --data-layout $data_layout
        # Otherwise, to optmize model with autoTVM and to log the results, run:
	#python test_resnet_inference_time.py --bit-config $bit_config --num-layers $num_layers --batch-size $batch_size --data-layout $data_layout --tuning-enable

	cp ./debug_output/resnet_generated.cu ./debug_output/resnet_manual.cu

	sed -i 's/h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 8;/h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 1;/g' ./debug_output/resnet_manual.cu
	sed -i 's/ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8;/ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1;/g' ./debug_output/resnet_manual.cu

	#sleep 5 
	python test_resnet_inference_time.py --bit-config $bit_config --num-layers $num_layers --batch-size $batch_size --data-layout $data_layout --manual-code
}

#run_inference "bit_config_resnet18_uniform8"        18
#run_inference "bit_config_resnet18_uniform4"        18
#run_inference "bit_config_resnet18_bops_0.75"       18
#run_inference "bit_config_resnet18_bops_0.5"        18
#run_inference "bit_config_resnet18_bops_0.25"       18
#run_inference "bit_config_resnet18_latency_0.75"    18
#run_inference "bit_config_resnet18_latency_0.5"     18
#run_inference "bit_config_resnet18_latency_0.25"    18
#run_inference "bit_config_resnet18_modelsize_0.75"  18
#run_inference "bit_config_resnet18_modelsize_0.5"   18
#run_inference "bit_config_resnet18_modelsize_0.25"  18

#run_inference "bit_config_resnet50_uniform8"        50 8 NHWC
#run_inference "bit_config_resnet50_uniform4"        50 8 NHWC
#run_inference "bit_config_resnet50_uniform8"        50 8 HWNC
run_inference "bit_config_resnet50_uniform4"        50 8 HWNC

#run_inference "bit_config_resnet50_bops_0.75"       50 
#run_inference "bit_config_resnet50_bops_0.5"        50 
#run_inference "bit_config_resnet50_bops_0.25"       50 
#run_inference "bit_config_resnet50_latency_0.75"    50 
#run_inference "bit_config_resnet50_latency_0.5"     50 
#run_inference "bit_config_resnet50_latency_0.25"    50 
#run_inference "bit_config_resnet50_modelsize_0.75"  50 
#run_inference "bit_config_resnet50_modelsize_0.5"   50 
#run_inference "bit_config_resnet50_modelsize_0.25"  50 
