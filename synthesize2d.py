# Palmprint synthesize in the 2D palne.
# The original implementation was written by Kai Zhao (kz@kaizhao.net) in Tencent, this is a reimplementation by Kai Zhao
# with confidential content removed.
# This reimplementation is only for research purpose, commercial use of this code must be officially permitted by Tencent.
# Copyright: Tencent
import bezier, mmcv
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

from multiprocessing import Pool

from vlkit.geometry.homography import random_perspective_matrix
from vlkit.image import norm255
from vlkit.utils import AverageMeter

import os, sys, argparse, glob, cv2, random, time
from os.path import join, split, isdir, isfile, dirname
from copy import copy
from tqdm import tqdm
from PIL import Image, ImageFilter, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description='Compare results')
    parser.add_argument('--num_ids', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--nproc', type=int, default=16)
    parser.add_argument('--imsize', type=int, default=256)
    parser.add_argument('--imagenet', type=str, default=None)
    parser.add_argument('--perspective', type=float, default=0, help='probability of performing perspective transform')
    parser.add_argument('--output', type=str, default='/data/generated_palm_prints')
    args = parser.parse_args()
    assert args.num_ids % args.nproc == 0
    return args


def wrap_points(points, M):
    assert isinstance(points, np.ndarray)
    assert isinstance(M, np.ndarray)
    n = points.shape[0]
    augmented_points = np.concatenate((points, np.ones((n, 1))), axis=1).astype(points.dtype)
    points = (M @ augmented_points.T).T
    points = points / points[:,-1].reshape(-1, 1)
    return points[:, :2]


def sample_edge(low, high):
    """
    sample points on edges of a unit square
    """
    offset = min(low, high)
    low, high = map(lambda x: x - offset, [low, high])
    t = np.random.uniform(low, high) + offset

    if t >= 4:
        t = t % 4
    if t < 0:
        t = t + 4

    if t <= 1:
        x, y = t, 0
    elif 1 < t <= 2:
        x, y = 1, t - 1
    elif 2 < t <= 3:
        x, y = 3 - t, 1
    else:
        x, y = 0, 4 - t
    return np.array([x, y]), t

def control_point(head, tail, t=0.5, s=0):
    head = np.array(head)
    tail = np.array(tail)
    l = np.sqrt(((head - tail) ** 2).sum())
    assert head.size == 2 and tail.size == 2
    assert l >= 0
    c = head * t + (1 - t) * tail
    x, y = head - tail
    v = np.array([-y, x])
    v /= max(np.sqrt((v ** 2).sum()), 1e-6)
    return c + s * l * v


def get_bezier(p0, p1, t=0.5, s=1):
    assert -1 < s < 1, 's=%f'%s
    c = control_point(p0, p1, t, s)
    nodes = np.vstack((p0, c, p1)).T
    return bezier.Curve(nodes, degree=2)


def generate_parameters():
    # head coordinates
    head1, thead1 = sample_edge(0, 0.2)
    head2, thead2 = sample_edge(-0.25, 0)
    head3, thead3 = sample_edge(-0.5, -0.2)
    head4, thead4 = sample_edge(1, 2)

    # tail coordinates
    tail1, ttail1 = sample_edge(1.2, 1.6)
    tail2, ttail2 = sample_edge(1.8, 2.25)
    tail3, ttail3 = sample_edge(2.3, 2.8)
    if thead4 >= 1.5:
        tail4, t = sample_edge(2.5, 3)
    else:
        tail4, t = sample_edge(2, 3)


    c1 = control_point(head1, tail1, s=np.random.uniform(0.13, 0.16))
    c2 = control_point(head2, tail2, s=-np.random.uniform(0.1, 0.2))
    c3 = control_point(head3, tail3, s=-np.random.uniform(0.1, 0.12))
    c4 = control_point(head4, tail4, s=-np.random.uniform(0.1, 0.12))

    return np.vstack((head1, c1, tail1)), np.vstack((head2, c2, tail2)), np.vstack((head3, c3, tail3)), np.vstack((head4, c4, tail4))


def batch_process(proc_index, ranges, args, imagenet_images=None):
    ids_per_proc = int(args.num_ids / args.nproc)
    EPS = 1e-2

    np.random.seed(proc_index)
    random.seed(proc_index)

    index_file = open(join(args.output, '%.3d-of-%.3d.txt' % (proc_index, args.nproc)), 'w')

    samples_per_proc = ids_per_proc * args.samples

    average_meter = AverageMeter(name='time')

    local_idx = 0
    for id_idx, i in enumerate(range(*ranges[proc_index])):

        tic = time.time()

        # start/end points of main creases
        nodes1 = generate_parameters()
        start1 = np.random.uniform(low=0, high=0.3, size=(len(nodes1))).tolist()
        end1 = np.random.uniform(low=0.7, high=1, size=(len(nodes1))).tolist()
        flag1 = [np.random.uniform()>0.01, np.random.uniform()>0.01, np.random.uniform()>0.01, np.random.uniform()>0.9]

        # start/end points of secondary creases
        n2 = np.random.randint(5, 15)
        coord2 = np.random.uniform(0, args.imsize, size=(n2, 2, 2))
        s2 = np.clip(np.random.normal(scale=0.4, size=(n2,)), -0.6, 0.6)
        t2 = np.clip(np.random.normal(loc=0.5, scale=0.4, size=(n2,)), 0.3, 0.7)

        # synthesize samples for each ID
        for s in range(args.samples):
            fig = plt.figure(frameon=False)
            canvas = fig.canvas
            dpi = fig.get_dpi()
            fig.set_size_inches((args.imsize + EPS) / dpi, (args.imsize + EPS) / dpi)
            # remove white edges by set subplot margin
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
            ax = plt.gca()
            ax.set_xlim(0, args.imsize)
            ax.set_ylim(args.imsize, 0)
            ax.axis('off')

            # determine the parameters of perspective transformations
            if np.random.uniform() < args.perspective:
                distortion_scale = np.random.uniform(0.01, 0.2)
                perspective_mat, (perspective_startpoints, perspective_endpoints), perspective_coeffs = \
                    random_perspective_matrix(args.imsize, args.imsize, distortion_scale=distortion_scale)
            else:
                perspective_mat = None

            global_idx = samples_per_proc * proc_index + local_idx
            if imagenet_images is not None:
                bg = imagenet_images[global_idx % len(imagenet_images)]
                bg_id = bg['label']
                bg_im = np.array(Image.open(bg['filename']).resize(size=(args.imsize,)*2))
                if np.random.uniform() >= 0.1:
                    kernel_size = (random.randint(0, 7) * 2 + 1,) * 2
                    bg_im = cv2.blur(bg_im, ksize=kernel_size)
            else:
                bg_im = np.random.normal(loc=0.0, size=(args.imsize, args.imsize, 3)) + np.random.uniform(size=(1, 1, 3))
                bg_im = norm255(np.clip(bg_im, 0, 1))
                bg_id = -1
                bg = {'filename': 'none'}
            bg_im = Image.fromarray(bg_im)

            ax.imshow(bg_im)

            # main creases
            curves1 = [bezier.Curve(n.T * args.imsize + np.random.uniform(-20, 20, size=n.T.shape), degree=2) for n in nodes1]
            points1 = [c.evaluate_multi(np.linspace(s, e, 50)).T for c, s, e in zip(curves1, start1, end1)]

            # perspective transformations
            if perspective_mat is not None:
                points1 = [wrap_points(p, perspective_mat) for p in points1]

            paths1 = [Path(p) for p in points1]
            lw1 = np.random.uniform(2.3, 2.7)
            patches1 =[patches.PathPatch(p, edgecolor=np.random.uniform(0, 0.4, 3), facecolor='none', lw=lw1) for p in paths1]
            for p, f in zip(patches1, flag1):
                if f:
                    ax.add_patch(p)

            # secondary creases
            # add turbulence to each sample
            coord2_ = coord2 + np.random.uniform(-5, 5, coord2.shape)
            s2_ = s2 + np.random.uniform(-0.1, 0.1, s2.shape)
            t2_ = t2 + np.random.uniform(-0.05, 0.05, s2.shape)

            lw2 = np.random.uniform(0.9, 1.1)
            for j in range(n2):
                points2 = get_bezier(coord2_[j, 0], coord2_[j, 1], t=t2_[j], s=s2_[j]).evaluate_multi(np.linspace(0, 1, 50)).T
                if perspective_mat is not None:
                    points2 = wrap_points(points2, perspective_mat)
                p = patches.PathPatch(Path(points2), edgecolor=np.random.uniform(0, 0.4, 3), facecolor='none', lw=lw2)
                ax.add_patch(p)

            stream, _ = canvas.print_to_buffer()
            buffer = np.frombuffer(stream, dtype='uint8')
            img_rgba = buffer.reshape(args.imsize, args.imsize, 4)
            rgb, alpha = np.split(img_rgba, [3], axis=2)
            img = rgb.astype('uint8')
            img = mmcv.rgb2bgr(img)

            if np.random.uniform() >= 0.2:
                kernel_size = (random.randint(0, 3) * 2 + 1,) * 2
                img = cv2.blur(img, ksize=kernel_size)

            filename = join(args.output, '%.5d' % i, '%.3d.jpg' % s)
            os.makedirs(dirname(filename), exist_ok=True)
            mmcv.imwrite(img, filename)
            plt.close()

            index_file.write('%s %d %d %s\n' % (join('%.5d' % i, '%.3d.jpg' % s), i, bg_id, bg['filename']))
            index_file.flush()

            local_idx += 1

        toc = time.time()
        average_meter.update(toc-tic)
        print("proc[%.3d/%.3d] id=%.4d [%.4d/%.4d]  (%.3f sec per id)" % (proc_index, args.nproc, i, id_idx, ids_per_proc, average_meter.avg))


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    spacing = np.linspace(0, args.num_ids,  args.nproc + 1).astype(int)

    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    if args.imagenet is not None:
        print('Loading imagenet images...')
        imagenet_images = []
        subfolders = [i for i in glob.glob('%s/train/n*' % args.imagenet) if isdir(i)]
        assert len(subfolders) == 1000, len(subfolders)
        for idx, d in enumerate(subfolders):
            imgs = glob.glob('%s/*.*' % d)
            imagenet_images.extend([{'filename': i, 'label': int(idx)} for i in imgs])
        print('%d images loaded, shuffling...' % len(imagenet_images))
        random.shuffle(imagenet_images)
        print('Done')
    else:
        imagenet_images = None

    argins = []
    for p in range(args.nproc):
        argins.append([p, ranges, args, imagenet_images])

    with Pool() as pool:
        pool.starmap(batch_process, argins)
