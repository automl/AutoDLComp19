

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
subFolder="ECOfullpy_UCF101_RGB_r1/"



pretrained_model="./pretrained_models/ECOfull_py_model.pth.tar"
#nECO_model=""
#############################################
#--- training hyperparams ---
dataset_name="ucf101"
netType="ECOfull_py"
batch_size=12
learning_rate=0.001
num_segments=16
consensus_type=identity #{avg, identity}
dropout=0.8
iter_size=5
num_workers=3

#####################################################################
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}/${subFolder}/training

echo "Current network folder: "
echo ${mainFolder}/${subFolder}


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

python3 -u main.py ${dataset_name} RGB  --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 4 --epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/ --consensus_type ${consensus_type} --eval-freq 1  --no_partialbn --freeze_eco --nesterov "True" --resume ${mainFolder}/${lastCheckpoint} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

else
     echo "Training with initialization"

python3 -u main.py ${dataset_name} RGB --arch ${netType} --num_segments ${num_segments} --gd 50 --lr ${learning_rate} --num_saturate 4 --epochs 40 -b ${batch_size} -i ${iter_size} -j ${num_workers} --dropout ${dropout} --snapshot_pref ${mainFolder}/${subFolder}/ --consensus_type ${consensus_type} --eval-freq 1  --no_partialbn --nesterov "True" --finetune_model ${pretrained_model} 2>&1 | tee -a ${mainFolder}/${subFolder}/training/log.txt

fi

#####################################################################


