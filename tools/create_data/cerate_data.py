# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os.path as osp
import random
import string

import cv2
import numpy as np
import tqdm
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Create data')
    parser.add_argument(
        '--ops', default=2, type=int, help='the operator numbers')
    parser.add_argument(
        '--dtype', default='train', type=str, help='the created data type')
    parser.add_argument(
        '--path', default='data/sft/', type=str, help='the save path for data')
    parser.add_argument(
        '--num', default=100, type=int, help='the numbers of data')
    parser.add_argument(
        '--max_num', default=20, type=int, help='the max operator value')
    parser.add_argument(
        '--name_length', default=20, type=int, help='the length of image name')
    args = parser.parse_args()
    return args


def create_image(path, name, formula, dtype):
    save_path = osp.join(path, dtype, f'{name}.jpg')
    img = np.zeros((224, 224), np.uint8)
    img[:, :] = 255
    cv2.imwrite(save_path, img)
    bk_img = cv2.imread(save_path)
    cv2.putText(bk_img, formula, (65, 112), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 0), 2)
    cv2.imwrite(save_path, bk_img)


def generate_random_string(length):
    """生成一个指定长度的随机字符串."""
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))


def create_formula(count, max_num):
    assert count in [2, 3]
    abc = [np.random.randint(1, max_num) for _ in range(count)]
    str_mat = ['+', '-', '*', '/']
    mat_index = [np.random.randint(0, 4) for _ in range(count - 1)]
    mat = [str_mat[i] for i in mat_index]
    if count == 2:
        formula = f'{abc[0]}{mat[0]}{abc[1]}'
    elif count == 3:
        formula = f'{abc[0]}{mat[0]}{abc[1]}{mat[1]}{abc[2]}'
    return eval(formula), formula + '='


def main():
    args = parse_args()
    num, length, dtype = args.num, args.name_length, args.dtype
    all_res = dict()

    for _ in tqdm.tqdm(list(range(num)), desc='create image'):
        # create different formula
        res, formula = create_formula(args.ops, args.max_num)
        if np.rint(res) != res:
            continue
        if res > 100 or res < -100:
            continue
        # create image with formula
        img_name = generate_random_string(length)
        # remove duplicate img name
        if img_name in all_res.keys():
            continue
        create_image(args.path, img_name, formula, dtype)
        all_res[img_name] = res
        logger.info(f'image name: {img_name}, formula: {formula}, res: {res}')

    logger.info(f'total number: {len(all_res.keys())}')
    # dump labels to json file
    json_path = osp.join(args.path, dtype, f'{dtype}_label.json')
    with open(json_path, 'w') as f:
        json.dump(all_res, f)


if __name__ == '__main__':
    main()
