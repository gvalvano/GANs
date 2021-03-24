# Flag for CUDA_VISIBLE_DEVICE:
CUDA_VD=0

for run_id_and_path in \
    'ACDC_UNet model_semi_unet' \
    'ACDC_UNet_Disc model_semi_unet_disc' \
    'ACDC_UNet_Disc_Textures model_semi_unet_disc_textures'

    do

    # shellcheck disable=SC2086
    set -- ${run_id_and_path}
    run_id=$1
    path=$2
    exp_type='semi'
    dpath='../DATA/ACDC'
    dset_name='acdc'
    table_name='SEMI_Experiments_ACDC'

    for perc in 'perc100' 'perc25' 'perc6'  # 'perc12p5' 'perc3'
        do for split in 'split0' 'split1' 'split2'

            do echo "${run_id}"_${perc}_${split}

            python -m train  --RUN_ID="${run_id}"_${perc}_${split} \
                             --n_epochs=300 \
                             --CUDA_VISIBLE_DEVICE=${CUDA_VD} \
                             --data_path=${dpath} \
                             --experiment_type=${exp_type} \
                             --experiment="${path}" \
                             --dataset_name=${dset_name} \
                             --notify=False \
                             --verbose=True \
                             --n_sup_vols=${perc} \
                             --split_number=${split}

            python -m test \
                            --RUN_ID="${run_id}"_${perc}_${split} \
                            --CUDA_VISIBLE_DEVICE=${CUDA_VD}  \
                            --data_path=${dpath} \
                            --experiment_type=${exp_type} \
                            --experiment="${path}" \
                            --dataset_name=${dset_name} \
                            --table_name="${table_name}" \
                            --notify=False \
                            --verbose=False \
                            --n_sup_vols=${perc} \
                            --split_number=${split} >> test_results.txt
            done
        done
    done

# ====================================================================================================

python -m notify --message="Train finished on GPU 0 (SEMI on ACDC)."