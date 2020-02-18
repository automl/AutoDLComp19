print('start import')
import tensorflow.compat.v2 as tf
print('import 1')
import tensorflow_datasets as tfds
print('import 2')
import multiprocessing as mp
print('import 3')
import os
print('import 4')
import sys
print('import 5')
import shutil
print('import 6')


def download_datasets(datasets, download_folder, process_id, num_processes):
        for i in range(len(datasets)):
                if (i + process_id) % num_processes != 0:
                        continue

                dataset = datasets[i]

                try:
                        tfds.load(name=dataset, data_dir=download_folder)
                except Exception as e:
                        shutil.rmtree(os.path.join(download_folder, dataset))
                        print(str(e))



def download_datasets_parallel(datasets, download_folder, num_processes):
    p_list = []
    for i in range(num_processes):
        p = mp.Process(
            target=download_datasets,
            args=(datasets, download_folder, i, num_processes)
        )
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()


if __name__ == "__main__":
    datasets_nlp = ['definite_pronoun_resolution', 'esnli',
        'glue/cola', 'glue/sst2', 'glue/mrpc', 'glue/qqp', 'glue/stsb', 'glue/mnli', 'glue/qnli',
        'glue/rte', 'glue/wnli', 'glue/ax',
        'imdb_reviews', 'movie_rationales', 'scicite', 'snli',
        'super_glue/boolq', 'super_glue/cb', 'super_glue/copa', 'super_glue/multirc', 'super_glue/record',
        'super_glue/rte', 'super_glue/wic', 'super_glue/wsc', 'super_glue/axb', 'super_glue/axg']

    download_folder_nlp = '/home/dingsda/tfds'
    #download_folder_nlp = '/data/aad/nlp_datasets/tfds'

    download_datasets_parallel(datasets_nlp, download_folder_nlp, 5)

