import os
import argparse
import img2dataset
from tqdm import tqdm
from concurrent import futures


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download LAION images of the LVIS/COCO novel categories')
    parser.add_argument('--name-file', default='promptdet_resources/lvis_category_and_description.txt',
                        help='LVIS/COCO category name and description')
    parser.add_argument('--base-ind-file', default='promptdet_resources/lvis_base_inds.txt',
                        help='index of the LVIS/COCO base categories')
    parser.add_argument('--base-category', action='store_true',
                        help='whether to retrieval the images of the base categories')
    parser.add_argument('--output-folder', default='data/laion_lvis/images',
                        help='output path')
    parser.add_argument('--num-thread', type=int, default=5,
                        help='the number of the thread to download the images')
    args = parser.parse_args()
    return args


def download_fun(cls_names, output_folder):
    for i, cls_name in tqdm(enumerate(cls_names), total=len(cls_names)):
        file_path = os.path.join(output_folder, cls_name + ".txt")
        image_path = os.path.join(output_folder, cls_name)
        img2dataset.download(url_list=file_path, image_size=1024, output_folder=image_path, processes_count=64, timeout=20)

    return True


def main():
    args = parse_args()

    output_folder = args.output_folder
    name_file = args.name_file
    base_ind_file = args.base_ind_file
    num_thread = args.num_thread

    lines = open(name_file).readlines()
    names = [line.strip().split(' ')[0] for line in lines]

    base_inds = open(base_ind_file, 'r').readline().strip().split(', ')
    base_inds = [int(ind) for ind in base_inds]
    novel_inds = [i for i in range(len(names)) if i not in base_inds]

    if args.base_category:
        names = [names[i] for i in base_inds]
    else:
        names = [names[i] for i in novel_inds]

    count_per_thread = (len(names) + num_thread - 1) // num_thread
    names_list = [names[i * count_per_thread:(i + 1) * count_per_thread] for i in range(num_thread)]

    with futures.ThreadPoolExecutor(max_workers=num_thread) as executor:
        threads = [executor.submit(download_fun, name, output_folder=output_folder) for name in names_list]
        for future in futures.as_completed(threads):
            print(future.result())


if __name__ == '__main__':
    main()
