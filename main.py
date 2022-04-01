import sys
sys.path.append(r"./Packages")

import math
import numpy as np
from numpy import linalg as lin
from PIL import Image


BLOCK_SIZE = 8
READ_PATH = "./Input/"
SAVE_PATH = "./Output/"
FORMAT = ".jpeg"

Q_50 = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                 [12, 12, 14, 19, 26, 58, 60, 55],
                 [14, 13, 16, 24, 40, 57, 69, 56],
                 [14, 17, 22, 29, 51, 87, 80, 62],
                 [18, 22, 37, 56, 68, 109, 103, 77],
                 [24, 35, 55, 64, 81, 104, 113, 92],
                 [49, 64, 78, 87, 103, 121, 120, 101],
                 [72, 92, 95, 98, 112, 100, 103, 99]])


def T_matrix():
    t = np.empty((BLOCK_SIZE, BLOCK_SIZE), dtype=float)

    # Compute the DCT matrix
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            if i == 0:
                t[i][j] = 1 / math.sqrt(BLOCK_SIZE)
            else:
                t[i][j] = math.sqrt(2 / BLOCK_SIZE) * \
                          math.cos(i * math.pi *(2 * j + 1) / (2 * BLOCK_SIZE))

    return t


def read_image(file: str):
    # read a jpeg or jpg
    im = Image.open(READ_PATH + file + FORMAT)

    # convert to black and white
    im.draft(mode="L", size=im.size)

    # put the gray scale value into numpy array (0 ~ 255)
    return np.array(im.getdata()).reshape(im.size[1], im.size[0])


def create_image(matrix: np.array, name: str):
    img = Image.new(mode="L", size=matrix.shape, color=255)

    # Create a jpeg image pixel by pixel from numpy array
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            img.putpixel((i, j), int(matrix[i][j]))

    img.save(SAVE_PATH + name + FORMAT)


def split_image(image: np.array, shape: tuple):
    rows = int(shape[0] / BLOCK_SIZE)
    cols = int(shape[1] / BLOCK_SIZE)

    # split image into 8x8 blocks
    blocks = list()
    for row in range(rows):
        for col in range(cols):
            block = np.empty((BLOCK_SIZE, BLOCK_SIZE), int)
            for i in range(BLOCK_SIZE):
                for j in range(BLOCK_SIZE):
                    block[i][j] = image[i + row * BLOCK_SIZE][j + col * BLOCK_SIZE]
            blocks.append(block)

    return blocks, (rows, cols)


def combine_image(blocks: list, shape: tuple):
    image = np.empty((BLOCK_SIZE * shape[0], BLOCK_SIZE * shape[1]), dtype=int)

    # Merge 8x8 blocks back to a complete image
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = i * shape[1] + j
            combine_helper(image, blocks[index], i, j)

    return image


def combine_helper(image: np.array, block: np.array, row: int, col: int):
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            image[i + BLOCK_SIZE * row][j + BLOCK_SIZE * col] = block[i][j]


def process_image(blocks: list, q_level: int):
    t = T_matrix()
    t_inv = lin.inv(t)
    q = Q_50

    # Calculate quantization matrix for a given level
    if q_level > 50:
        q = np.round(q * (100 - q_level) / 50).astype(int).clip(1, 255)
    elif q_level < 50:
        q = np.round(q * 50 / q_level).astype(int).clip(1, 255)

    # Process each block individually
    for i in range(len(blocks)):
        blocks[i] = process_block(blocks[i], t, t_inv, q)

    return blocks


def process_block(block: np.array, t: np.array, t_inv: np.array, q: np.array):
    # Discrete cosine transformation
    d = t.dot(block - 128).dot(t_inv)

    # Quantization
    c = np.round(d / q).astype(int)

    # Decompression
    n = np.round(t_inv.dot(q * c).dot(t)).astype(int) + 128

    return n


def compress(input_name, output_name, quality):
    img = read_image(input_name)
    blocks, shape = split_image(img, img.shape)
    new_img = combine_image(process_image(blocks, int(quality)), shape)
    create_image(new_img.transpose(), output_name)


def main():
    print("—————————————————————————————————————————————————————————")
    print("                 JPEG Compressor                         ")
    print("   1. Enter file names without format extension          ")
    print("   2. Enter an integer quality level between 1 ~ 100     ")
    print("—————————————————————————————————————————————————————————")
    print()

    input_name = input("Enter input file: ")
    quality = input("Enter quality level: ")
    output_name = input("Enter output file: ")
    compress(input_name, output_name, quality)

    print()
    print("—————————————————————————————————————————————————————————")
    print("                  Compress Complete                      ")
    print("—————————————————————————————————————————————————————————")


if __name__ == '__main__':
    main()
