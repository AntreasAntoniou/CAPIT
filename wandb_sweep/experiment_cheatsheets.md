Currently running on tpu-large
```bash
python run.py hydra.verbose=True \
resume=True \
batch_size=32 \
datamodule.num_workers=-1 \
trainer.gpus=0 \
trainer.tpu_cores=8 \
+trainer.strategy=ddp \
+trainer.sync_batchnorm=True \
model=milli_modus_prime_vi-transformer16 \
datamodule=debug-tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=False \
datamodule.dataset_config.modality_config.video=False \
datamodule.dataset_config.rescan_paths=False \
datamodule.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=base
```



Currently running on gpu-instance-0
```bash
python run.py hydra.verbose=False \
resume=True \
batch_size=1000 \
datamodule.num_workers=-1 \
trainer.gpus=-1 \
model=centi_modus_prime_resnet50 \
datamodule=tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=False \
datamodule.dataset_config.modality_config.video=False \
datamodule.dataset_config.rescan_paths=False \
datamodule.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=deci
```

Currently running on gpu-instance-1
```bash
python run.py hydra.verbose=False \
resume=True \
batch_size=1000 \
datamodule.num_workers=-1 \
trainer.gpus=-1 \
model=deci_modus_prime_vi-transformer16 \
datamodule=tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=False \
datamodule.dataset_config.modality_config.video=False \
datamodule.dataset_config.rescan_paths=False \
datamodule.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=deci
```

Currently running on gpu-instance-2
```bash
python run.py hydra.verbose=False \
resume=False \
batch_size=1000 \
datamodule.num_workers=32 \
trainer.gpus=-1 \
model=deci_modus_prime_vi-transformer16 \
datamodule=tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=True \
datamodule.dataset_config.modality_config.video=True \
datamodule.dataset_config.rescan_paths=False \
datamodule.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=deci
```

Currently running on gpu-instance-3
```bash
python run.py hydra.verbose=False \
resume=True \
batch_size=1000 \
datamodule.num_workers=-1 \
trainer.gpus=-1 \
model=centi_modus_prime_resnet50 \
datamodule=tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=False \
datamodule.dataset_config.modality_config.audio=True \
datamodule.dataset_config.modality_config.video=False \
datamodule.dataset_config.rescan_paths=False \
datamodule.prefetch_factor=1 \
datamodule.dataset_config.dataset_size_identifier=deci
```

# Currently running on gpu-medium-instance-0
```bash
python run.py \
hydra.verbose=False \
resume=False \
batch_size=250 \
datamodule.num_workers=16 \
trainer.gpus=1 \
model=base_modus_prime_vi-transformer16 \
datamodule=tali \
datamodule.dataset_config.modality_config.image=True \
datamodule.dataset_config.modality_config.text=True \
datamodule.dataset_config.modality_config.audio=False \
datamodule.dataset_config.modality_config.video=False \
datamodule.dataset_config.rescan_paths=True \
datamodule.prefetch_factor=2 \
datamodule.dataset_config.dataset_size_identifier=base \
model/lr_scheduler_config=reduce_lr_on_plateau \
seed=20130023 \
model.optimizer_config.lr=0.00001
```

```bash
gcloud beta compute instances create gpu-instance-small-0 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-highgpu-8g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=8,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=gpu-instance-large-0,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-v-1-3,mode=rw,size=150,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-standard \
--create-disk=auto-delete=yes,device-name=tali-dataset-disk,image=projects/tali-multi-modal/global/images/tali-v-3-5-high-npy-error-rate,mode=rw,name=disk-6,size=10000,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-ssd \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--provisioning-model=SPOT
```

```bash
gcloud beta compute instances create gpu-instance-large-0 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-highgpu-8g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--preemptible \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=8,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=gpu-instance-large-0,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-v-1-3,mode=rw,size=150,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-standard \
--create-disk=auto-delete=yes,device-name=tali-dataset-disk,image=projects/tali-multi-modal/global/images/tali-v-3-5-high-npy-error-rate,mode=rw,name=disk-6,size=10000,type=projects/tali-multi-modal/zones/us-central1-f/diskTypes/pd-ssd \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--provisioning-model=SPOT
```


