import os
import torch
import argparse
import faiss
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description='Retrieval LAION images of the LVIS/COCO categories')
    parser.add_argument('--indice-folder', help='LAION400M indice')
    parser.add_argument('--metadata', help='LAION400M metadata')
    parser.add_argument('--text-features', default='promptdet_resources/lvis_category_embeddings.pt',
                        help='LVIS/COCO category embeddings')
    parser.add_argument('--name-file', default='promptdet_resources/lvis_category_and_description.txt',
                        help='LVIS/COCO category name and description')
    parser.add_argument('--base-ind-file', default='promptdet_resources/lvis_base_inds.txt',
                        help='index of the LVIS/COCO base categories')
    parser.add_argument('--base-category', action='store_true',
                        help='whether to retrieval the images of the base categories')
    parser.add_argument('--output-folder', default='data/laion_lvis/images',
                        help='output path')
    parser.add_argument('--num-images', type=int, default=500,
                        help='the number of sourced images per catogory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    indice_folder = args.indice_folder
    metadata = args.metadata
    text_features = torch.load(args.text_features)
    name_file = args.name_file
    base_ind_file = args.base_ind_file
    output_folder = args.output_folder
    num_images = args.num_images

    os.mkdir(output_folder)

    h = h5py.File(metadata, 'r')
    df = h['dataset']
    url_list = None
    if "url" in df:
        url_list = df["url"]

    image_list = df["url"]
    image_index = faiss.read_index(indice_folder+"/image.index")

    lines = open(name_file).readlines()
    names = [line.strip().split(' ')[0] for line in lines]

    base_inds = open(base_ind_file, 'r').readline().strip().split(', ')
    base_inds = [int(ind) for ind in base_inds]
    novel_inds = [i for i in range(len(names)) if i not in base_inds]

    if args.base_category:
        names = [names[i] for i in base_inds]
        text_features = text_features[base_inds]
    else:
        names = [names[i] for i in novel_inds]
        text_features = text_features[novel_inds]

    for index, name in enumerate(names):
        text_feature = text_features[index:index + 1]
        index = image_index

        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        query = text_feature.cpu().detach().numpy().astype("float32")

        D, I = index.search(query, num_images)
        I = I[0]
        D = D[0]

        min_D = min(D)
        max_D = max(D)

        print(f"The minimum distance is {min_D:.2f} and the maximum is {max_D:.2f}")

        url_results = []
        for _, i in zip(D, I):
            if i >= len(image_list):
                continue
            if url_list is not None:
                line = url_list[i].decode('UTF-8') + '\n'
                if line not in url_results:
                    url_results.append(line)

        file_path = os.path.join(output_folder, name + ".txt")
        print(f"result is saved in {file_path}")
        f = open(file_path, 'w', encoding='utf-8')
        f.writelines(url_results)
        f.close()


if __name__ == '__main__':
    main()
