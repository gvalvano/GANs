# Flag for CUDA_VISIBLE_DEVICE:
CUDA_VD=0

dpath='DATA/cityscapes'
res_dir='.'
dset_name='cityscapes'
table_name='SEMI_Experiments_CS'
EPOCHS=1000


for perc in '8vols' '5vols' '6vols' '3vols'
    do for split in 'split0' 'split1' 'split2'
    do

    path="exp_gan"

    for run_id_and_gan_type in \
        'NSatGAN nsatgan' \
        'LSGAN lsgan' \
        'WGAN wgan'
        do

        # shellcheck disable=SC2086
        set -- ${run_id_and_gan_type}
        run_id=$1
        gan_type=$2
        exp_type='semi'

        r_id="${run_id}"_${perc}_${split}
        echo "${r_id}"
        python -m train --RUN_ID="${r_id}" \
                        --n_epochs=${EPOCHS} \
                        --use_spectral_norm=y \
                        --use_instance_noise=y \
                        --use_label_smoothing=n \
                        --CUDA_VISIBLE_DEVICE=${CUDA_VD} \
                        --data_path=${dpath} \
                        --experiment_type=${exp_type} \
                        --experiment="${path}" \
                        --dataset_name=${dset_name} \
                        --notify=n \
                        --verbose=y \
                        --n_sup_vols=${perc} \
                        --split_number=${split} \
                        --table_name=${table_name} \
                        --results_dir=${res_dir} \
                        --gan="${gan_type}"
        done
    done
done

# ====================================================================================================

python -m notify --message="Train finished on GPU ${CUDA_VD} (SEMI on Cityscapes)."
