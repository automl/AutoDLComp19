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
#--- training hyperparams ---
#'jhmdb21','jester','somethingv2','hmdb51','kinetics','epickitchen_verb','epickitchen_noun','yfcc100m'
dataset_name="jhmdb21"
netType="resnet50"
batch_size=1 #43
num_segments=8
consensus_type=avg #{avg, identity}
iter_size=4 # batch_size * iter_size = pseudo_batch_size 
num_workers=32
optimizer="SGD"
val_perc=0.02
#############################################
#--- bohb hyperparams ---
bohb_iterations=10
min_budget=0.01
max_budget=0.1
eta=3
bohb_workers=1
#############################################
#--- finetunint hyperparams --- 
# Set training = True and params, to run finetune instead of BOHB
training=False
pretrained_model="pretrained_models/somethingv2_rgb_epoch_16_checkpoint.pth.tar"
dropout=0.8
learning_rate=0.001
epochs=80
#############################################
# Folders
mainFolder="experiments/"
subFolder="run_${netType}_${dataset_name}_${optimizer}_finetune_${finetune}_r1/"
mkdir -p ${mainFolder}
mkdir -p ${mainFolder}${subFolder}training
echo "Current network folder "
echo ${mainFolder}${subFolder}
snapshot_pref="${mainFolder}${subFolder}${netType}_${dataset_name}_${optimizer}_finetune_${finetune}"
#############################################
# others
print=True
#####################################################################
#####################################################################
# Find the latest checkpoint of network
checkpointIter="$(ls ${mainFolder}${subFolder}*checkpoint* 2>/dev/null | grep -o "epoch_[0-9]*_" | sed -e "s/^epoch_//" -e "s/_$//" | xargs printf "%d\n" | sort -V | tail -1 | sed -e "s/^0*//")"
echo "${checkpointIter}"
#####################################################################
if [ "x${checkpointIter}" != "x" ]; then
    lastCheckpoint="${subFolder}${snap_pref}_rgb_epoch_${checkpointIter}_checkpoint.pth.tar"
    echo "Continuing from checkpoint ${lastCheckpoint}"

    python3 -u main.py \
    --dataset ${dataset_name} \
    --modality RGB \
    --arch ${netType} \
    --resume ${mainFolder}${lastCheckpoint} \
    --num_segments ${num_segments} \
    --gd 50 \
    --training ${training}? \
    --lr ${learning_rate} --num_saturate 4 \
    --dropout ${dropout} \
    --epochs ${epochs} \
    --val_perc ${val_perc} \
    -b ${batch_size} \
    -i ${iter_size} \
    -j ${num_workers} \
    --snapshot_pref ${mainFolder}${subFolder} \
    --shift --shift_div=8 --shift_place=blockres --dense_sample \
    --consensus_type ${consensus_type} \
    --eval-freq 1 \
    --no_partialbn \
    --freeze_eco \
    --freeze_interval 2 50 0 0 \
    --nesterov "True" \
    --optimizer ${optimizer} \
    --bohb_iterations ${bohb_iterations} \
    --min_budget ${min_budget} \
    --max_budget ${max_budget} \
    --eta ${eta} \
    --bohb_workers ${bohb_workers}  \
    --snapshot_pref ${snapshot_pref} \
    --working_directory ${mainFolder}${subFolder} \
    2>&1 | tee -a ${mainFolder}${subFolder}training/log.txt

else
     echo "Training with initialization"

    python3 -u main.py \
    --dataset ${dataset_name} \
    --modality RGB \
    --arch ${netType} \
    --finetune_model ${pretrained_model} \
    --num_segments ${num_segments} \
    --gd 50 \
    --training ${training}? \
    --lr ${learning_rate} --num_saturate 4 \
    --dropout ${dropout} \
    --epochs ${epochs} \
    --val_perc ${val_perc} \
    -b ${batch_size} \
    -i ${iter_size} \
    -j ${num_workers} \
    --optimizer ${optimizer} \
    --nesterov "True" \
    --snapshot_pref ${mainFolder}${subFolder} \
    --shift --shift_div=8 --shift_place=blockres --dense_sample \
    --consensus_type ${consensus_type} \
    --eval-freq 1 \
    --no_partialbn \
    --freeze_eco \
    --freeze_interval 2 50 0 0 \
    --bohb_iterations ${bohb_iterations} \
    --bohb_workers ${bohb_workers}  \
    --min_budget ${min_budget} \
    --max_budget ${max_budget} \
    --eta ${eta} \
    --snapshot_pref ${snapshot_pref} \
    --working_directory ${mainFolder}${subFolder} \
    2>&1 | tee -a ${mainFolder}${subFolder}training/log.txt

fi
#####################################################################
