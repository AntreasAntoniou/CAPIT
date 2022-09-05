gcloud compute instances create tali-sweep-8g-node-template-1 \
--project=tali-multi-modal \
--zone=us-central1-f \
--machine-type=a2-highgpu-8g \
--network-interface=network-tier=PREMIUM,subnet=default \
--no-restart-on-failure \
--maintenance-policy=TERMINATE \
--provisioning-model=SPOT \
--service-account=tali-multi-modal@tali-multi-modal.iam.gserviceaccount.com \
--scopes=https://www.googleapis.com/auth/cloud-platform \
--accelerator=count=8,type=nvidia-tesla-a100 \
--create-disk=auto-delete=yes,boot=yes,device-name=tali-sweep-8g-template,image=projects/tali-multi-modal/global/images/tali-ubuntu-cuda110-pytorch-v-1-2,mode=rw,size=50,type=projects/tali-multi-modal/zones/us-central1-a/diskTypes/pd-standard \
--create-disk=auto-delete=yes,device-name=persistent-disk-1,image=projects/tali-multi-modal/global/images/tali-dataset-v3-2-us-central1-full,mode=rw,size=3500,type=projects/tali-multi-modal/zones/us-central1-a/diskTypes/pd-ssd \
--no-shielded-secure-boot \
--shielded-vtpm \
--shielded-integrity-monitoring \
--reservation-affinity=any \
--metadata=startup-script=setup_scripts/startup_call.sh

