### ResNet50

- base + image + text = 2g for 64 batch size
- base + image + text + audio = 4g for 64 batch size
- base + image + text + audio + video = 8g for 64 batch size

- centi + image + text = 1g for 64 batch size (can probably fit 256)
- centi + image + text + audio = 1g for 64 batch size
- centi + image + text + audio + video = 2g for 64 batch size (70% memory utilized can probably fit 96)
- centi + image + text + video = 2g for 48

### ViTransformer16

- base + image + text = 2g for 64 batch size
- base + image + text + audio = 4g for 64 batch size
- base + image + text + audio + video = 8g for 64 batch size

- centi + image + text = 1g for 64 batch size (can probably fit 256)
- centi + image + text + audio = 1g for 64 batch size <-
- centi + image + text + audio + video = 2g for 64 batch size 


````
/mnt/disk/tali/experiments//TALI-gcp-sweep-1-milli-tali-centi_modus_prime_resnet50-video-False-audio-True-text-True-image-True-20221601//checkpoints/
````

Write script that generates scripts that include the target command in the startup script of an instance, and also starts said instance. This is probably by far the easiest and most well controlled way of doing what we need without many bells or whistles.


###########################################################################Updates

### ResNet50

- centi + image + text(77) + video(10) = 2g for 32 👍
- centi + image + video(10) = 2g for 32 👍
- centi + image + text(77) = 2g for 512 👍

- deci + image + text(77) + video(10) = 2g for 16 👍
- deci + image + video(10) = 2g for 16 👍
- deci + image + text(77) = 2g for 160 👍

- base + image + text(77) + video(10) = 2g for 16 👍
- base + image + video(10) = 2g for 48 👍
- base + image + text(77) = 2g for 512 👍

