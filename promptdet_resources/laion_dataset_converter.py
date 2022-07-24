import os
import os.path as osp
import mmcv
import argparse
from tqdm import tqdm
from tools.promptdet.class_names import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Laion images of the LVIS/COCO novel categories to mmdetection format')
    parser.add_argument('--data-path', help='data path') # data_root
    parser.add_argument('--out-file', help='output path')
    parser.add_argument('--base-ind-file', default='promptdet_resources/lvis_base_inds.txt',
                        help='index of the LVIS/COCO base categories')
    parser.add_argument('--topK', default=300, help='the number of images per catogory')
    args = parser.parse_args()
    return args

def file_filter(f):
    if f[-4:] in ['.jpg', '.png', 'bmp']:
        return True
    else:
        return False

def main():
    args = parse_args()

    data_root = args.data_path
    out_file = args.out_file
    base_ind_file = args.base_ind_file
    topK_images = args.topK

    base_inds = open(base_ind_file, 'r').readline().strip().split(', ')
    base_inds = [int(ind) for ind in base_inds]

    annotations = []
    images = []
    obj_count = 0
    img_id = 0
    number_class_save = 0
    for category_id, dir_name in tqdm(enumerate(LVIS_CLASSES), total=len(LVIS_CLASSES)):
        image_prefix = osp.join(data_root, dir_name, '00000')
        if category_id in base_inds:
            continue

        number_class_save += 1
        filenames = os.listdir(image_prefix)
        filenames = list(filter(file_filter, filenames))

        filenames = sorted(filenames)[:topK_images]
        print(f"#images of class {dir_name}: {len(filenames)}")

        for filename in filenames:
            img_path = osp.join(image_prefix, filename)
            height, width = mmcv.imread(img_path).shape[:2]

            images.append(dict(
                id=img_id,
                file_name=osp.join(dir_name, '00000', filename),
                height=height,
                width=width))

            data_anno = dict(
                image_id=img_id,
                id=obj_count + 1,
                category_id=category_id + 1,
                bbox=[0, 0, 1, 1], # not used, only for compatibility with mmdetection dataloder
                area=1,
                iscrowd=0)
            annotations.append(data_anno)

            obj_count += 1
            img_id += 1

    categories = []
    for idx, cls_name in enumerate(LVIS_CLASSES):
        categories.append(dict(
            id=idx + 1,
            name=cls_name
        ))

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=categories)
    mmcv.dump(coco_format_json, out_file)

    print(f'#novel categories: {number_class_save}')


if __name__ == '__main__':
    main()
