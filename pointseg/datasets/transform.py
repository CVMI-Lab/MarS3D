import random
import scipy
import scipy.ndimage
import scipy.interpolate
import numpy as np
import torch

from pointseg.utils.registry import Registry

TRANSFORMS = Registry("transforms")

@TRANSFORMS.register_module()
class Collect(object):
    def __init__(self,
                 keys,
                 feat_keys,
                 meta_keys=()):
        self.keys = keys
        self.feat_keys = feat_keys
        self.meta_keys = meta_keys

    def __call__(self, data_dict):
        data_metas = {}
        for key in self.meta_keys:
            data_metas[key] = data_dict[key]
        feat = torch.cat([data_dict[key].float() for key in self.feat_keys], dim=1)
        data = dict(data_metas=data_metas, feat=feat)
        for key in self.keys:
            data[key] = data_dict[key]
        
        return data


@TRANSFORMS.register_module()
class ToTensor(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = torch.from_numpy(data_dict["coord"])
            if not isinstance(data_dict["coord"], torch.FloatTensor):
                data_dict["coord"] = data_dict["coord"].float()

        if "color" in data_dict.keys():
            data_dict["color"] = torch.from_numpy(data_dict["color"])
            if not isinstance(data_dict["color"], torch.FloatTensor):
                data_dict["color"] = data_dict["color"].float()

        if "norm" in data_dict.keys():
            data_dict["norm"] = torch.from_numpy(data_dict["norm"])
            if not isinstance(data_dict["norm"], torch.FloatTensor):
                data_dict["norm"] = data_dict["norm"].float()

        if "label" in data_dict.keys():
            data_dict["label"] = torch.from_numpy(data_dict["label"])
            if not isinstance(data_dict["label"], torch.LongTensor):
                data_dict["label"] = data_dict["label"].long()

        if "label_b" in data_dict.keys():
            data_dict["label_b"] = torch.from_numpy(data_dict["label_b"])
            if not isinstance(data_dict["label_b"], torch.LongTensor):
                data_dict["label_b"] = data_dict["label_b"].long()
                
        if "label_c" in data_dict.keys():
            data_dict["label_c"] = torch.from_numpy(data_dict["label_c"])
            if not isinstance(data_dict["label_c"], torch.LongTensor):
                data_dict["label_c"] = data_dict["label_c"].long()
                
        if "weight" in data_dict.keys():
            data_dict["weight"] = torch.from_numpy(data_dict["weight"])
            if not isinstance(data_dict["weight"], torch.FloatTensor):
                data_dict["weight"] = data_dict["weight"].float()

        if "index" in data_dict.keys():
            data_dict["index"] = torch.from_numpy(data_dict["index"])
            if not isinstance(data_dict["index"], torch.LongTensor):
                data_dict["index"] = data_dict["index"].long()

        if "discrete_coord" in data_dict.keys():
            data_dict["discrete_coord"] = torch.from_numpy(data_dict["discrete_coord"])
            if not isinstance(data_dict["discrete_coord"], torch.IntTensor):
                data_dict["discrete_coord"] = data_dict["discrete_coord"].int()

        if "inverse" in data_dict.keys():
            data_dict["inverse"] = torch.from_numpy(data_dict["inverse"])
            if not isinstance(data_dict["inverse"], torch.LongTensor):
                data_dict["inverse"] = data_dict["inverse"].long()

        if "length" in data_dict.keys():
            data_dict["length"] = torch.from_numpy(data_dict["length"])
            if not isinstance(data_dict["length"], torch.LongTensor):
                data_dict["length"] = data_dict["length"].long()

        if "be_input" in data_dict.keys():
            data_dict["be_input"] = torch.from_numpy(data_dict["be_input"])
            if not isinstance(data_dict["be_input"], torch.LongTensor):
                data_dict["be_input"] = data_dict["be_input"].long()
        
        if "main_label" in data_dict.keys():
            data_dict["main_label"] = torch.from_numpy(data_dict["main_label"])
            if not isinstance(data_dict["main_label"], torch.LongTensor):
                data_dict["main_label"] = data_dict["main_label"].long()
        
        return data_dict


@TRANSFORMS.register_module()
class NormalizeColor(object):
    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict


@TRANSFORMS.register_module()
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


@TRANSFORMS.register_module()
class PositiveShift(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            coord_min = np.min(data_dict["coord"], 0)
            data_dict["coord"] -= coord_min
        return data_dict


@TRANSFORMS.register_module()
class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


@TRANSFORMS.register_module()
class PointClip(object):
    def __init__(self, point_cloud_range=(-80, -80, -3, 80, 80, 1)):
        self.point_cloud_range = point_cloud_range

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            data_dict["coord"] = np.clip(data_dict["coord"],
                                         a_min=self.point_cloud_range[:3],
                                         a_max=self.point_cloud_range[3:])
        return data_dict


@TRANSFORMS.register_module()
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
            upright_axis: axis index among x,y,z, i.e. 2 for z
            """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "norm" in data_dict.keys():
                data_dict["norm"] = data_dict["norm"][idx]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict


@TRANSFORMS.register_module()
class RandomShift(object):
    def __init__(self, shift=None):
        self.shift = shift if shift is not None else [0.2, 0.2, 0]

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            shift_x = np.random.uniform(-self.shift[0], self.shift[0])
            shift_y = np.random.uniform(-self.shift[1], self.shift[1])
            shift_z = np.random.uniform(-self.shift[2], self.shift[2])
            data_dict["coord"] += [shift_x, shift_y, shift_z]
        return data_dict


@TRANSFORMS.register_module()
class RandomRotate(object):
    def __init__(self,
                 angle=None,
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "norm" in data_dict.keys():
            data_dict["norm"] = np.dot(data_dict["norm"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateTargetAngle(object):
    def __init__(self,
                 angle=(1/2, 1, 3/2),
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.75):
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "norm" in data_dict.keys():
            data_dict["norm"] = np.dot(data_dict["norm"], np.transpose(rot_t))
        return data_dict


@TRANSFORMS.register_module()
class RandomScale(object):
    def __init__(self, scale=None, anisotropic=False):
        self.scale = scale if scale is not None else [0.95, 1.05]
        self.anisotropic = anisotropic

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
            data_dict["coord"] *= scale
        return data_dict


@TRANSFORMS.register_module()
class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if "norm" in data_dict.keys():
                data_dict["norm"][:, 0] = -data_dict["norm"][:, 0]
        if np.random.rand() < self.p:
            if "coord" in data_dict.keys():
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
            if "norm" in data_dict.keys():
                data_dict["norm"][:, 1] = -data_dict["norm"][:, 1]
        return data_dict


@TRANSFORMS.register_module()
class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        assert (self.clip > 0)
        if "coord" in data_dict.keys():
            jitter = np.clip(self.sigma * np.random.randn(data_dict["coord"].shape[0], 3), -self.clip, self.clip)
            data_dict["coord"] += jitter
        return data_dict


@TRANSFORMS.register_module()
class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][:, :3] + blend_factor * contrast_feat
        return data_dict


@TRANSFORMS.register_module()
class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() < self.p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= self.std * 255
            data_dict["color"][:, :3] = np.clip(noise + data_dict["color"][:, :3], 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, data_dict):
        if "color" in data_dict.keys():
            # Assume color[:, :3] is rgb
            hsv = HueSaturationTranslation.rgb_to_hsv(data_dict["color"][:, :3])
            hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
            sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
            hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
            hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
            data_dict["color"][:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return data_dict


@TRANSFORMS.register_module()
class RandomDropColor(object):
    def __init__(self, p=0.8, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment

    def __call__(self, data_dict):
        if "color" in data_dict.keys() and np.random.rand() > self.p:
            data_dict["color"] *= self.color_augment
        return data_dict

    def __repr__(self):
        return 'RandomDropColor(color_augment: {}, p: {})'.format(self.color_augment, self.p)


@TRANSFORMS.register_module()
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(data_dict["coord"], granularity, magnitude)
        return data_dict


@TRANSFORMS.register_module()
class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 return_inverse=False,
                 return_discrete_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.return_inverse = return_inverse
        self.return_discrete_coord = return_discrete_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(np.int)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        if self.mode == 'train':  # train mode
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_unique = idx_sort[idx_select]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_unique]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_unique]
            if "norm" in data_dict.keys():
                data_dict["norm"] = data_dict["norm"][idx_unique]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx_unique] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
            if "label_b" in data_dict.keys():
                data_dict["label_b"] = data_dict["label_b"][idx_unique] \
                    if len(data_dict["label_b"]) != 1 else data_dict["label_b"]
            if "label_c" in data_dict.keys():
                data_dict["label_c"] = data_dict["label_c"][idx_unique] \
                    if len(data_dict["label_c"]) != 1 else data_dict["label_c"]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if True:
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
                data_dict["length"] = np.array(inverse.shape)
                data_dict["count"] = np.array(len(count))
            return data_dict

        elif self.mode == 'test':  # test mode
            idx_data = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                idx_data.append(idx_part)
            return data_dict, idx_data
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


@TRANSFORMS.register_module()
class SphereCrop(object):
    def __init__(self, point_max=80000, mode="random"):
        self.point_max = point_max
        assert mode in ["random", "center", "all"]
        self.mode = mode

    def __call__(self, data_dict, idx=None):
        assert "coord" in data_dict.keys()
        if self.mode == "all":
            # TODO: Optimize
            if idx is None:
                idx = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            # coord_list, color_list, dist2_list, idx_list, offset_list = [], [], [], [], []
            if data_dict["coord"].shape[0] > self.point_max:
                coord_p, idx_uni = np.random.rand(data_dict["coord"].shape[0]) * 1e-3, np.array([])
                while idx_uni.size != idx.shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2), 1)
                    idx_crop = np.argsort(dist2)[:self.point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "norm" in data_dict.keys():
                        data_crop_dict["norm"] = data_dict["norm"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = idx[idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"]))
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(np.concatenate((idx_uni, data_crop_dict["index"])))
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = idx
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > self.point_max:
            if self.mode == "random":
                center = data_dict["coord"][np.random.randint(data_dict["coord"].shape[0])]
            elif self.mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[:self.point_max]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "norm" in data_dict.keys():
                data_dict["norm"] = data_dict["norm"][idx_crop]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx_crop] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict



@TRANSFORMS.register_module()
class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        shuffle_index = np.arange(data_dict["coord"].shape[0])
        np.random.shuffle(shuffle_index)
        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "norm" in data_dict.keys():
            data_dict["norm"] = data_dict["norm"][shuffle_index]
        if "label" in data_dict.keys():
            data_dict["label"] = data_dict["label"][shuffle_index] \
                if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict


class Compose(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.transforms = []
        for t_cfg in self.cfg:
            self.transforms.append(TRANSFORMS.build(t_cfg))

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


