
#!/bin/bash


run_profile() {
	bit_config=$1
	num_layers=$2
    batch_size=$3
    data_layout=$4

	printf "========%s========\n" $bit_config

    if [ $1 = "bit_config_resnet50_uniform8" ]
    then
        precision="int8"
    else
        precision="int4"
    fi
	
    #generate autogen cu code
	python test_resnet_inference_time.py --bit-config $bit_config --num-layers $num_layers --batch-size $batch_size --data-layout $data_layout

	cp ./debug_output/resnet_generated.cu ./debug_output/resnet_manual.cu

    #generate maual code
	sed -i 's/h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 8;/h_w_fused_n_fused_i_fused_nn_fused_ii_fused_inner < 1;/g' ./debug_output/resnet_manual.cu
	sed -i 's/ax0_ax1_fused_ax2_fused_ax3_fused_inner < 8;/ax0_ax1_fused_ax2_fused_ax3_fused_inner < 1;/g' ./debug_output/resnet_manual.cu

	sleep 5 
    ncu --force-overwrite --section "ComputeWorkloadAnalysis|InstructionStats|LaunchStats|MemoryWorkloadAnalysis|MemoryWorkloadAnalysis_Chart|MemoryWorkloadAnalysis_Tables|Occupancy|SchedulerStats|SpeedOfLight|SpeedOfLight_RooflineChart|WarpStateStats" --sampling-max-passes 16 --profile-from-start 0 --export profile/resnet50_batch_${batch_size}_${precision}_${data_layout} python test_resnet_inference_time.py --bit-config $bit_config --num-layers $num_layers --batch-size $batch_size --data-layout $data_layout --manual-code
}



#run_profile "bit_config_resnet50_uniform8"        50 8 NHWC
#run_profile "bit_config_resnet50_uniform4"        50 8 NHWC
#run_profile "bit_config_resnet50_uniform8"        50 8 HWNC
run_profile "bit_config_resnet50_uniform4"        50 8 HWNC
