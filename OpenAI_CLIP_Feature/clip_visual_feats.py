from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import visual_extractor
import basic_utils as utils
from params import parse_args


def custom_path2id(path):
    return int(os.path.basename(path).split('.')[0].split('_')[-1])


def main(args):
    # load image list
    imgs = os.listdir(args.image_dir)
    # build src_list & dst_list
    src_list=[]
    dst_list=[]
    for img in imgs:
        if os.path.exists(os.path.join(args.output_dir, img.split('.jpg')[0])+'.npz'):
            continue
        src_list.append(os.path.join(args.image_dir, img))
        dst_list.append(os.path.join(args.output_dir, img.split('.jpg')[0]))
    src_list, dst_list = utils.shuffle_list(src_list, dst_list)

    # extract
    worker = visual_extractor.create(args.ve_name, args, src_list, dst_list)
    worker.extract()


if __name__ == "__main__":
    args = parse_args()
    if not args.debug:
        utils.mkdirp(args.output_dir)
    main(args)
