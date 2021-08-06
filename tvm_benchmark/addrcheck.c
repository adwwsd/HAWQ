





((unsigned int*)T_cast)[((((((((((int)blockIdx.z) * 1024) + (ax2_outer_inner * 128)) + (ax2_inner_ax3_inner_fused_outer * 64)) + ((((int).x) >> 3) * 16)) + (((int)blockIdx.y) * 8)) + (((int).z) * 2)) + ax3_outer_inner))] = ((unsigned int)((((((((0 | ((
max(min(((int)((((((long)max((Conv[(ADDR1)] + ((int*)placeholder2)[(ADDR2)]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 28)) | ((
max(min(((int)((((((long)max((Conv[((ADDR1 + 1))] + ((int*)placeholder2)[((ADDR2 + 1))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 24)) | ((
max(min(((int)((((((long)max((Conv[((ADDR1 + 2))] + ((int*)placeholder2)[((ADDR2 + 2))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 20)) | ((
max(min(((int)((((((long)max((Conv[((ADDR1 + 3))] + ((int*)placeholder2)[((ADDR2 + 3))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 16)) | ((
max(min(((int)((((((long)max((Conv[((ADDR1 + 4))] + ((int*)placeholder2)[((ADDR2 + 4))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 12)) | ((
max(min(((int)((((((long)max((Conv[((ADDR1 + 5))] + ((int*)placeholder2)[((ADDR2 + 5))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 8)) | ((
max(min(((int)((((((long)max((Conv[((ADDR1 + 6))] + ((int*)placeholder2)[((ADDR2 + 6))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15) << 4)) | (
max(min(((int)((((((long)max((Conv[((ADDR1 + 7))] + ((int*)placeholder2)[((ADDR2 + 7))]), 0)) << (long)4) * (long)1241513984) + (long)1073741824) >> (long)31)), 15), 0) & 15)));






acc = 0
for i in [0,7]:
    biasedConv = (Conv[((ADDR1 + i))] + ((int*)placeholder2)[((ADDR2 + i))])
    relued = ((long)max(biasedConv, 0))

    val1 = ((relued << (long)4) * (long)37 * 2^25) + (long)2^30
    val2 = (int)(val1 >> (long)31)

    clipped = max(min(val2, 15),0)
    maksed = clipped & 15

    acc |= maksed << 2^(7-i)

destination[ADDR] = acc
