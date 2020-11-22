import numpy as np
import cv2
import torch
from Utils.transformations import get_transform


def get_multi_scale_size(image, input_size, current_scale, min_scale):
    h, w, _ = image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63)//64 * 64)
    # scale is relative to min scale and min_input_size
    if w < h:
        w_resized = int(min_input_size * current_scale / min_scale)
        h_resized = int(
            int((min_input_size/w*h+63)//64*64)*current_scale/min_scale
        )
        scale_w = w / 200.0
        scale_h = h_resized / w_resized * w / 200.0
    else:
        h_resized = int(min_input_size * current_scale / min_scale)
        w_resized = int(
            int((min_input_size/h*w+63)//64*64)*current_scale/min_scale
        )
        scale_h = h / 200.0
        scale_w = w_resized / h_resized * h / 200.0

    return (w_resized, h_resized), center, np.array([scale_w, scale_h])


def get_multi_scale_size_hourglass(image, input_size, currect_scale, min_scale):
    h, w, _ = image.shape
    center = np.array([w / 2.0, h / 2.0])
    scale = max(h, w) / 200

    inp_res = int((currect_scale * 512 + 63) // 64 * 64)

    return (inp_res, inp_res), center, np.array([scale, scale])


def resize_align_multi_scale(image, input_size, current_scale, min_scale):
    size_resized, center, scale = get_multi_scale_size(
        image, input_size, current_scale, min_scale
    )
    trans = _get_affine_transform(center, scale, 0, size_resized)

    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized
        # (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale

def resize_align_multi_scale_hourglass(image, input_size, current_scale, min_scale):
    size_resized, center, scale = get_multi_scale_size_hourglass(
        image, input_size, current_scale, min_scale
    )
    trans = get_transform(center, scale, size_resized, 0)[:2]

    image_resized = cv2.warpAffine(
        image,
        trans,
        size_resized
        # (int(w_resized), int(h_resized))
    )

    return image_resized, center, scale

def _get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])
    # overall goal is to compute two sets of 3 points which are related by an affine transform
    scale_tmp = scale * 200.0  # effectively
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift  # scale_tmp is used to scale the shift s.t. same shift parameters can
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def aggregate_results(
        cfg, scale_factor, final_heatmaps, tags_list, heatmaps, tags
):
    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list


def aggregate_results_mpn(
        cfg, scale_factor, final_heatmaps, tags_list, final_features, heatmaps, tags, features
):

    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]
    if len(features) == 2:
        raise NotImplementedError
    else:
        assert len(features) == 1
        features_avg = features[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    if final_features is None:
        final_features = features_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_features += features_avg
    else:
        final_features += torch.nn.functional.interpolate(
            features_avg,
            size=(final_features.size(2), final_features.size(3)),
            mode='bilinear',
            align_corners=False
        )


    return final_heatmaps, tags_list, final_features


def aggregate_results_mpn_hourglass(
        cfg, scale_factor, final_heatmaps, tags_list, final_features, heatmaps, tags, features
):

    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]
    if len(features) == 2:
        raise NotImplementedError
    else:
        assert len(features) == 1
        features_avg = features[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    if final_features is None:
        final_features = features_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_features += features_avg
    else:
        final_features += torch.nn.functional.interpolate(
            features_avg,
            size=(final_features.size(2), final_features.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list, final_features


def aggregate_results_hourglass(
        cfg, scale_factor, final_heatmaps, tags_list, heatmaps, tags
):

    if scale_factor == 1 or len(cfg.TEST.SCALE_FACTOR) == 1:
        if final_heatmaps is not None and not cfg.TEST.PROJECT2IMAGE:
            tags = [
                torch.nn.functional.interpolate(
                    tms,
                    size=(final_heatmaps.size(2), final_heatmaps.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
                for tms in tags
            ]
        for tms in tags:
            tags_list.append(torch.unsqueeze(tms, dim=4))

    heatmaps_avg = (heatmaps[0] + heatmaps[1])/2.0 if cfg.TEST.FLIP_TEST \
        else heatmaps[0]

    if final_heatmaps is None:
        final_heatmaps = heatmaps_avg
    elif cfg.TEST.PROJECT2IMAGE:
        final_heatmaps += heatmaps_avg
    else:
        final_heatmaps += torch.nn.functional.interpolate(
            heatmaps_avg,
            size=(final_heatmaps.size(2), final_heatmaps.size(3)),
            mode='bilinear',
            align_corners=False
        )

    return final_heatmaps, tags_list


def get_final_preds(grouped_joints, center, scale, heatmap_size):
    final_results = []
    for person in grouped_joints[0]:
        joints = np.zeros((person.shape[0], 3))
        joints = _transform_preds(person, center, scale, heatmap_size)
        final_results.append(joints)

    return final_results

def _transform_preds(coords, center, scale, output_size):
    # target_coords = np.zeros(coords.shape)
    target_coords = coords.copy()
    trans = _get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = _affine_transform(coords[p, 0:2], trans)
    return target_coords


def _affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def _get_3rd_point(a, b):
    # direct: is direction from b to a
    # output: 2d cross product resulting in an orthogonal vector to ba
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    # src_point is rotated by rot_rad
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def multiscale_keypoints(keypoints, factors, image, input_size, scale, min_scale, project_to_image):

    def _affine_joints(joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def _affine_factors(factors, mat):
        return factors * mat[0, 0] * mat[1, 1]

    resized_img, center, scale = get_multi_scale_size(image, input_size, scale, min_scale)

    factor = 1 if project_to_image else 0.5
    target_shape = (int(resized_img[0] * factor), int(resized_img[1] * factor))
    mat = _get_affine_transform(center, scale, 0, target_shape)
    keypoints[0, :, :, :2] = _affine_joints(keypoints[0, :, :, :2], mat)
    factors = _affine_factors(factors, mat)

    return keypoints, factors


def multiscale_keypoints_hourglass(keypoints, factors, image, input_size, scale, min_scale, project_to_image):

    def _affine_joints(joints, mat):
        joints = np.array(joints)
        shape = joints.shape
        joints = joints.reshape(-1, 2)
        return np.dot(np.concatenate(
            (joints, joints[:, 0:1]*0+1), axis=1), mat.T).reshape(shape)

    def _affine_factors(factors, mat):
        return factors * mat[0, 0] * mat[1, 1]

    resized_img, center, scale = get_multi_scale_size_hourglass(image, input_size, scale, min_scale)

    factor = 1 if project_to_image else 0.25
    target_shape = (int(resized_img[0] * factor), int(resized_img[1] * factor))
    mat = _get_affine_transform(center, scale, 0, target_shape)
    keypoints[0, :, :, :2] = _affine_joints(keypoints[0, :, :, :2], mat)
    factors = _affine_factors(factors, mat)

    return keypoints, factors

FLIP_CONFIG = {
    'COCO': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
    'COCO_WITHOUT_REARANGING': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ],
    'COCO_WITH_CENTER': [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 17
    ],
    'CROWDPOSE': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13
    ],
    'CROWDPOSE_WITH_CENTER': [
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12, 13, 14
    ]
}
