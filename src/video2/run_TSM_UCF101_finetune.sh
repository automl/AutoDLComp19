
unset CUDA_VISIBLE_DEVICES
gpuName="$(nvidia-smi --query-gpu="gpu_name" --format=csv,noheader,nounits -i 0)"
echo "$(date) - Training started on host ${HOSTNAME} on an ${gpuName}"


#####################################################################
# Train on local machine
if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
    cd $PBS_O_WORKDIR
fi


#####################################################################
# Parameters!
mainFolder="/AutoDLComp19/src/video2/experiments/"
subFolder="TSM_UCF101_RGB_r1/"



pretrained_model="/pretrained_models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth"
#############################################
#--- training hyperparams ---
dataset_name="ucf101"
netType="resnet50"
batch_size=8 #43
learning_rate=0.001
num_segments=8
consensus_type=avg #{avg, identity}

dropout=0.8
iter_size=4
num_workers=1

#####################################################################
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}/training

echo "Current network folder "
Echo ${mainFolder}/${subFolder}


#####################################################################
# Find the latest checkpoint of network
checkpointIter="$(ls ${mainFolder}/${subFolder}/*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
#####################################################################


echo "${checkpointIter}"

#####################################################################
# If there is a checkpoint then continue training otherwise train from scratch
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}/${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

python3 -u main.py ${dataset_name} RGB  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 4 --epochs 80 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder} --shift --shift_div=8 --shift_place=blockres --dense_sample --consensus_type ${consensus_type} --eval-freq 1  --no_partialbn --nesterov "True" --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

else
     echo "Training with initialization"

python3 -u main.py ${dataset_name} RGB --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 4 --epochs 80 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder} --shift --shift_div=8 --shift_place=blockres --dense_sample --consensus_type ${consensus_type} --eval-freq 1  --no_partialbn --nesterov "True" --finetune_model ${pretrained_model} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

fi

#####################################################################


