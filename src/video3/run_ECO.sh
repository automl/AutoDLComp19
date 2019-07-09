#unset CUDA_VISIBLE_DEVICES
#gpuName="$(nvidia-smi --query-gpu="gpu_name" --format=csv,noheader,nounits -i 0)"
#echo "$(date) - Training started on host ${HOSTNAME} on an ${gpuName}"
#####################################################################
# Train on local machine
#if [ "$1" != "local" ] && [ "$2" != "local" ] && [ "$3" != "local" ]; then
#    cd $PBS_O_WORKDIR
#fi
#####################################################################
# Parameters!
#############################################
#--- bohb hyperparams ---
bohb_iterations=1
min_budget=0.001
max_budget=0.001
eta=3
bohb_workers=1
val_perc=0.0001
#############################################
#--- training hyperparams ---
dataset_name="somethingv2"
netType="ECOfull"
batch_size=2 #43
num_segments=8
consensus_type=avg #{avg, identity}
iter_size=4
num_workers=28
optimizer="SGD"
#############################################
#--- finetunint hyperparams --- 
# Set finetune = True and params, to run finetune instead of BOHB
finetune=True
pretrained_model="pretrained_models/eco_fc_rgb_kinetics.pth.tar"
dropout=0.8
learning_rate=0.001
#############################################
# Folders
mainFolder="experiments/"
subFolder="run_ECO_${dataset_name}_${optimizer}_finetune_${finetune}_r1/"
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}${subFolder}training
echo "Current network folder "
Echo ${mainFolder}${subFolder}
#############################################
# others
print=False
#####################################################################
#####################################################################
# Find the latest checkpoint of network
checkpointIter="$(ls ${mainFolder}${subFolder}*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
echo "${checkpointIter}"
#####################################################################
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

    python3 -u main.py ${dataset_name} RGB \
    --arch ${netType} \
    --num_segments ${num_segments} \
    --gd 50 \
    --lr ${learning_rate} --num_saturate 4 \
    --epochs 80 \
    -b ${batch_size} \
    -i ${iter_size} \
    -j ${num_workers} \
    --dropout ${dropout} \
    --snapshot_pref ${mainFolder}${subFolder} \
    --consensus_type ${consensus_type} \
    --eval-freq 1 \
    --no_partialbn \
    --freeze_eco \
    --freeze_interval 2 50 0 0 \
    --nesterov "True" \
    --resume ${mainFolder}${lastCheckpoint} \
    --working_directory ${mainFolder} \
    --optimizer ${optimizer} \
    --bohb_iterations ${bohb_iterations} \
    --min_budget ${min_budget} \
    --max_budget ${max_budget} \
    --eta ${eta} \
    --val_perc ${val_perc} \
    --bohb_workers ${bohb_workers}  \
    2>&1 | tee -a ${mainFolder}${subFolder}training/log.txt

else
     echo "Training with initialization"

    python3 -u main.py ${dataset_name} RGB \
    --arch ${netType} \
    --num_segments ${num_segments} \
    --gd 50 \
    --lr ${learning_rate} --num_saturate 4 \
    --epochs 80 \
    -b ${batch_size} \
    -i ${iter_size} \
    -j ${num_workers} \
    --dropout ${dropout} \
    --snapshot_pref ${mainFolder}${subFolder} \
    --consensus_type ${consensus_type} \
    --eval-freq 1 \
    --no_partialbn \
    --freeze_eco \
    --freeze_interval 2 50 0 0 \
    --nesterov "True" \
    --finetune_model ${pretrained_model} \
    --working_directory ${mainFolder}${subFolder} \
    --optimizer ${optimizer} \
    --bohb_iterations ${bohb_iterations} \
    --bohb_workers ${bohb_workers}  \
    --min_budget ${min_budget} \
    --max_budget ${max_budget} \
    --eta ${eta} \
    --val_perc ${val_perc} \
    2>&1 | tee -a ${mainFolder}${subFolder}training/log.txt

fi
#####################################################################
