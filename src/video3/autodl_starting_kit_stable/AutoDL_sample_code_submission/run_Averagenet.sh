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
#'ECOfull_py'','resnet50','resnet101','ECO','ECOfull'
netType="Averagenet"
batch_size=4 #32 for 4 2080/titan X
num_segments=16
consensus_type=avg #{avg, identity}
iter_size=1 # batch_size * iter_size = pseudo_batch_size 
num_workers=32 #10 for 4 2080/ 32 for 4 titan X
optimizer="SGD"
val_perc=0.5
class_limit=10000
#############################################
#--- bohb hyperparams ---
bohb_iterations=10
min_budget=0.05
max_budget=0.1
eta=2
bohb_workers=1
#############################################
#--- finetunint hyperparams --- 
# Set training = True and params, to run finetune instead of BOHB
training=False
finetune_model='./pretrained_models/Averagenet_RGB_Kinetics_128.pth.tar'
dropout=0.2
learning_rate=0.00001
epochs=20
#############################################
#--- Multilabel hyperparams --- 
prediction_threshold=0.7
#############################################
# Folders
mainFolder="experiments/"
subFolder="run_${netType}_${dataset_name}_${optimizer}_finetune_${training}_r1/"
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
    --training ${training} \
    --lr ${learning_rate} --num_saturate 3 \
    --dropout ${dropout} \
    --epochs ${epochs} \
    --val_perc ${val_perc} \
    --class_limit ${class_limit} \
    -b ${batch_size} \
    -i ${iter_size} \
    -j ${num_workers} \
    --snapshot_pref ${mainFolder}${subFolder} \
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
    --prediction_threshold ${prediction_threshold} \
    2>&1 | tee -a ${mainFolder}${subFolder}training/log.txt

else
     echo "Training with initialization"

    python3 -u main.py \
    --dataset ${dataset_name} \
    --modality RGB \
    --arch ${netType} \
    --finetune_model ${finetune_model} \
    --num_segments ${num_segments} \
    --gd 50 \
    --training ${training} \
    --lr ${learning_rate} --num_saturate 4 \
    --dropout ${dropout} \
    --epochs ${epochs} \
    --val_perc ${val_perc} \
	--class_limit ${class_limit} \
    -b ${batch_size} \
    -i ${iter_size} \
    -j ${num_workers} \
    --optimizer ${optimizer} \
    --nesterov "True" \
    --snapshot_pref ${mainFolder}${subFolder} \
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
    --prediction_threshold ${prediction_threshold} \
    2>&1 | tee -a ${mainFolder}${subFolder}training/log.txt

fi
#####################################################################
