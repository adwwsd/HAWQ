
print("global to global check")
i=0
for j in range(0,1024):
    dst = ((((i) * 1024) + (j))) 
    src = (((((int(((((((i) * 4) + ((j) >> 8)) % 10) * 16) + ((j) & 15)) / 3) * 192) + (int((((i) * 4) + ((j) >> 8)) / 10) * 48)) + ((((j) & 255) >> 4) * 3)) + (((((((i) * 4) + ((j) >> 8)) % 10) * 16) + ((j) & 15)) % 3)))
    print(f"dst:{dst}, src:{src}")



print("global to shared check")
i=0
for tx in range(0,32):
    for ty in range(0,8):
        for tz in range(0,2):
            for k in range(0,5):
                for l in range(0,8):
                    dst =(((((((ty) * 2560) + ((tz) * 1280)) + (k * 256)) + ((tx) * 8)) + l))
                    src = (((((int(((((i) * 128) + ((ty) * 16)) + ((((tx) * 8) + l) >> 4)) / 12544) * 158700) + (int((((((i) * 128) + ((ty) * 16)) + ((((tx) * 8) + l) >> 4)) % 12544) / 112) * 1380)) + (int(((((tz) * 80) + (k * 16)) + ((((tx) * 8) + l) & 15)) / 21) * 690)) + ((((((i) * 128) + ((ty) * 16)) + ((((tx) * 8) + l) >> 4)) % 112) * 6)) + (((((tz) * 80) + (k * 16)) + ((((tx) * 8) + l) & 15)) % 21))
                    print(f"dst:{dst}, src:{src}")



