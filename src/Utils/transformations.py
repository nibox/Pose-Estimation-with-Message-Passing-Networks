import numpy as np
import torchvision
import torch


def reverse_affine_map(keypoints, img_size_orig, scaling_type):
    """
    Reverses the transformation resulting from the input argument (using get_transform). Used to map output keypoints to
    original space in order to evaluate them.
    :param keypoints:
    :param img_size_orig: (Width, Height) of the source image.
    :param scaling_type: Type of scaling that was performed on the image/keypoints.
            "short": Short side of the image was resized to target size preserving the original aspect ratio
            while not adding black borders. Width_new_image != height_new_image
            "short_long": Width_new_image == height_new_image. Black borders are added to preserve the aspect ratio of
            the content
    :return: keypoints with respect to original image coordinates
    """
    if scaling_type == "short":
        resized_img, center, scale = get_multi_scale_size(img_size_orig[1], img_size_orig[0], 512, 1., 1.)
        mat = get_transform(center, scale, (int(resized_img[0]/2), int(resized_img[1]/2)))

    elif scaling_type == "short_long":
        gt_width = img_size_orig[0]
        gt_height = img_size_orig[1]
        scale = max(gt_height, gt_width) / 200
        scale = np.array([scale, scale])
        # resized_img, center, scale = get_multi_scale_size(img_size_orig[1], img_size_orig[0], 512, 1., 1.)
        # mat = get_transform(center, scale, (int(resized_img[0] / 2), int(resized_img[1] / 2)))
        mat = get_transform((gt_width / 2, gt_height / 2), scale, (128, 128))
    else:
        raise NotImplementedError

    inv_mat = np.zeros([3, 3], dtype=np.float)
    inv_mat[0, 0], inv_mat[1, 1] = 1 / mat[0, 0], 1 / mat[1, 1]
    inv_mat[0, 2], inv_mat[1, 2] = -mat[0, 2] / mat[0, 0], -mat[1, 2] / mat[1, 1]
    inv_mat = inv_mat[:2]
    inv_mat = np.linalg.inv(mat)[:2]  # might this lead to numerical errors?
    keypoints[:, :, :2] = kpt_affine(keypoints[:, :, :2], inv_mat)
    return keypoints


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def kpt_affine(kpt, mat):
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    return np.dot(np.concatenate((kpt, kpt[:, 0:1] * 0 + 1), axis=1), mat.T).reshape(shape)


def factor_affine(factors, mat):
    return factors * mat[0, 0] * mat[1, 1]


def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h[1]
    t[1, 1] = float(res[0]) / h[0]
    t[0, 2] = res[1] * (-float(center[0]) / h[0] + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h[1] + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def get_multi_scale_size(img_h, img_w, input_size, current_scale, min_scale):
    h, w = img_h, img_w  # image.shape
    center = np.array([int(w / 2.0 + 0.5), int(h / 2.0 + 0.5)])

    # calculate the size for min_scale
    min_input_size = int((min_scale * input_size + 63)//64 * 64)
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