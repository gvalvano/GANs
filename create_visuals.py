import numpy as np
import os
from idas.data_utils.png_utils import get_png_image
from glob import glob
from PIL import Image
from idas.utils import safe_mkdir

root_dir = './results'
dataset = ["acdc", "cap", "chaos_T1", "chaos_T2", "ppss"]
line_spacing = 2


for dset in dataset:
    print('Dataset: {0}'.format(dset))
    image_dir = os.path.join(root_dir, '{0}/images'.format(dset))
    subdir_list = glob(os.path.join(image_dir, '*'))

    for subdir in subdir_list:
        perc = subdir.rsplit('/')[-1]
        run_ids = [el.rsplit('/')[-1].rsplit('_{0}'.format(perc))[0]
                   for el in glob(os.path.join(subdir, '*')) if el.endswith('split0')]
        # get image size:
        sizes = run_ids[0].rsplit('_')[-1]
        h, w = [int(sizes.rsplit('x')[0]), int(sizes.rsplit('x')[1])]
        h = 160 if dset == 'ppss' and h == 168 else h  # correct typo in name

        # get run id for weak and semi supervision, remove image size:
        run_ids_no_size = [el[:-len(el.rsplit('_')[-1])-1] for el in run_ids]  # removes image size from run id
        semi_run_ids = [el for el in run_ids_no_size if el.lower().startswith('semi')]
        weak_run_ids = [el for el in run_ids_no_size if el.lower().startswith('weak')]

        # read figures names for each experiment:
        path = os.path.join(image_dir, perc)
        path = os.path.join(path, '{0}_{1}_split0/*'.format(run_ids[0], perc))
        figures_names = [el.rsplit('/')[-1] for el in glob(path)]

        for split in ['split0', 'split1', 'split2']:
            for experiment, ids in zip(['semi', 'weak'], [semi_run_ids, weak_run_ids]):
                # for the given percentage, for each experiment type (semi/weak), each split, and across run_id:

                # ---------------------------------------------------------------------------------------------------
                # load input image, ground truth and predicted segmentation from each experiment:
                images = dict()
                images['input'] = []
                images['ground_truth'] = []
                images['prediction'] = dict()

                for k, r_id in enumerate(ids):
                    images['prediction'][r_id] = []

                    for name in figures_names:
                        fname = '{0}/{1}/{2}_{3}_{1}_{4}/{5}'.format(image_dir, perc, r_id, sizes, split, name)
                        figure = get_png_image(fname)
                        for line in range(figure.shape[0] // h):
                            if k == 0:
                                images['input'].append(figure[line * h: (line+1) * h, :w])
                                images['ground_truth'].append(figure[line * h: (line+1) * h, -w:])
                            images['prediction'][r_id].append(figure[line * h: (line + 1) * h, w: -w])

                # ---------------------------------------------------------------------------------------------------
                # begin new figure:

                # compute figure size:
                n_columns = len(ids) + 2  # one for each run id + input image + ground truth segmentation
                n_rows = len(images['input'])
                height = n_rows * h + line_spacing * (n_rows - 1)
                width = n_columns * w + line_spacing * (n_columns - 1)

                # initialize empty array:
                panel = np.zeros((height, width, 3), dtype=np.uint8)  # RGB, 3 channels
                for r in range(n_rows):
                    panel[r * (h + line_spacing): (r + 1) * h + r * line_spacing, :w, :] = images['input'][r]
                    panel[r * (h + line_spacing): (r + 1) * h + r * line_spacing,
                          w + line_spacing: 2 * w + line_spacing, :] = images['ground_truth'][r]

                    for k, r_id in enumerate(ids):
                        c = k + 2
                        panel[r * (h + line_spacing): (r + 1) * h + r * line_spacing,
                              c * (w + line_spacing): (c + 1) * w + c * line_spacing, :] = images['prediction'][r_id][r]

                # ---------------------------------------------------------------------------------------------------
                # save image
                panel_name = '{0}_panel_{1}_{2}.png'.format(experiment, perc, split)
                report = '1) img\n2) gt\n'
                for ii, r_id in enumerate(ids):
                    report += '{0}) {1}\n'.format(ii, r_id.lower().rsplit(dset)[-1])

                dest_dir = os.path.join(root_dir, 'IMAGES')
                safe_mkdir(dest_dir)
                dest_dir = os.path.join(dest_dir, dset)
                safe_mkdir(dest_dir)
                dest_dir = os.path.join(dest_dir, experiment)
                safe_mkdir(dest_dir)

                im = Image.fromarray(panel)
                im.save(os.path.join(dest_dir, panel_name))

                with open(os.path.join(dest_dir, 'report.txt'), 'w') as f:
                    f.write(report)

print('\nDone.')
