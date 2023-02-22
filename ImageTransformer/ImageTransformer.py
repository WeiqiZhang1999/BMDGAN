#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import numpy as np
from scipy import ndimage
from PIL import ImageEnhance
from PIL import Image as pil_image
from typing import Optional
from Utils.ImageHelper import ImageHelper


class IdentityTransformer:

    def __init__(self):
        pass

    def get_random_transform(self, img_shape: tuple[int, int], seed: Optional[int] = None, *args, **kwargs):
        return {}

    def apply_transform(self, x: np.ndarray, transform_parameters: dict[str, any], *args, **kwargs):
        return x


class ImageTransformer:

    def summary(self):
        message = "ImageTransformer------------------------\n"
        message += "rotation_range: {}\n".format(self.rotation_range)
        message += "width_shift_range: {}\n".format(self.width_shift_range)
        message += "height_shift_range: {}\n".format(self.height_shift_range)
        message += "shear_range: {}\n".format(self.shear_range)
        message += "zoom_range: {}\n".format(self.zoom_range)
        message += "lock_zoom_ratio: {}\n".format(self.lock_zoom_ratio)
        message += "channel_shift_range: {}\n".format(self.channel_shift_range)
        message += "fill_mode: {}\n".format(self.fill_mode)
        message += "cval: {}\n".format(self.cval)
        message += "horizontal_flip: {}\n".format(self.horizontal_flip)
        message += "vertical_flip: {}\n".format(self.vertical_flip)
        message += "dtype: {}\n".format(self.dtype)
        message += "interpolation_order: {}\n".format(self.interpolation_order)
        message += "End of ImageTransformer------------------------\n"
        return message

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 lock_zoom_ratio=True,
                 channel_shift_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 data_format='channels_last',
                 interpolation_order=1,
                 dtype="float64",
                 brightness_range=None,
                 contrast_range=None,
                 ):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.lock_zoom_ratio = lock_zoom_ratio
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.dtype = dtype
        self.interpolation_order = interpolation_order

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError(
                '`data_format` should be `"channels_last"` '
                '(channel after row and column) or '
                '`"channels_first"` (channel before row and column). '
                'Received: %s' % data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
        if data_format == 'channels_last':
            self.channel_axis = 3
            self.row_axis = 1
            self.col_axis = 2

        if isinstance(zoom_range, (float, int)):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif (len(zoom_range) == 2 and
              all(isinstance(val, (float, int)) for val in zoom_range)):
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received: %s' % (zoom_range,))

        if brightness_range is not None:
            if (not isinstance(brightness_range, (tuple, list)) or
                    len(brightness_range) != 2):
                raise ValueError(
                    '`brightness_range should be tuple or list of two floats. '
                    'Received: %s' % (brightness_range,))
        self.brightness_range = brightness_range

        if contrast_range is not None:
            if (not isinstance(contrast_range, (tuple, list)) or
                    len(contrast_range) != 2):
                raise ValueError(
                    '`contrast_range should be tuple or list of two floats. '
                    'Received: %s' % (contrast_range,))
        self.contrast_range = contrast_range

    def get_random_transform(self, img_shape: tuple[int, int], seed: Optional[int] = None):
        """Generates random parameters for a transformation.

        # Arguments
            seed: Random seed.
            img_shape: Tuple of integers.
                Shape of the image that is transformed.

        # Returns
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(
                -self.rotation_range,
                self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:  # 1-D array-like or int
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:  # 1-D array-like or int
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:  # floating point
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(
                -self.shear_range,
                self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 1)[0]
            if self.lock_zoom_ratio:
                zy = zx
            else:
                zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 1)[0]

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.random() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                        self.channel_shift_range)

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(self.brightness_range[0],
                                           self.brightness_range[1])

        contrast = None
        if self.contrast_range is not None:
            contrast = np.random.uniform(self.contrast_range[0],
                                           self.contrast_range[1])

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness,
                                'contrast': contrast,
                                }

        return transform_parameters

    def apply_transform(self, x: np.ndarray, transform_parameters: dict[str, any],
                        fill_mode=None, cval=None, interpolation_order=None):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intensity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        img_channel_axis = self.channel_axis - 1

        x = ImageTransformer.apply_affine_transform(x, transform_parameters.get('theta', 0),
                                                    transform_parameters.get('tx', 0),
                                                    transform_parameters.get('ty', 0),
                                                    transform_parameters.get('shear', 0),
                                                    transform_parameters.get('zx', 1),
                                                    transform_parameters.get('zy', 1),
                                                    row_axis=img_row_axis,
                                                    col_axis=img_col_axis,
                                                    channel_axis=img_channel_axis,
                                                    fill_mode=self.fill_mode if fill_mode is None else fill_mode,
                                                    cval=self.cval if cval is None else cval,
                                                    order=self.interpolation_order if interpolation_order is None else interpolation_order)

        if transform_parameters.get('channel_shift_intensity') is not None:
            x = ImageTransformer.apply_channel_shift(x,
                                                     transform_parameters['channel_shift_intensity'],
                                                     img_channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = ImageTransformer.flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = ImageTransformer.flip_axis(x, img_row_axis)

        if transform_parameters.get('brightness') is not None:
            x = ImageHelper.brightness(x, transform_parameters['brightness'])
        #     x = ImageHelper.apply_brightness_shift(x, transform_parameters['brightness'])

        if transform_parameters.get('contrast') is not None:
            x = ImageHelper.contrast(x, transform_parameters['contrast'])

        return x

    @staticmethod
    def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                               row_axis=0, col_axis=1, channel_axis=2,
                               fill_mode='nearest', cval=0., order=1):
        """Applies an affine transformation specified by the parameters given.

        # Arguments
            x: 2D numpy array, single image.
            theta: Rotation angle in degrees.
            tx: Width shift.
            ty: Heigh shift.
            shear: Shear angle in degrees.
            zx: Zoom in x direction.
            zy: Zoom in y direction
            row_axis: Index of axis for rows in the input image.
            col_axis: Index of axis for columns in the input image.
            channel_axis: Index of axis for channels in the input image.
            fill_mode: Points outside the boundaries of the input
                are filled according to the given mode
                (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
            cval: Value used for points outside the boundaries
                of the input if `mode='constant'`.
            order: int, order of interpolation

        # Returns
            The transformed version of the input.
        """
        transform_matrix = None
        if theta != 0:
            theta = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shift_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear = np.deg2rad(shear)
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                     [0, np.cos(shear), 0],
                                     [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = shear_matrix
            else:
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            if transform_matrix is None:
                transform_matrix = zoom_matrix
            else:
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[row_axis], x.shape[col_axis]
            transform_matrix = ImageTransformer.transform_matrix_offset_center(
                transform_matrix, h, w)
            x = np.rollaxis(x, channel_axis, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]

            channel_images = [ndimage.interpolation.affine_transform(
                x_channel,
                final_affine_matrix,
                final_offset,
                order=order,
                mode=fill_mode,
                cval=cval) for x_channel in x]
            x = np.stack(channel_images, axis=0)
            x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    @staticmethod
    def apply_channel_shift(x, intensity, channel_axis=0):
        """Performs a channel shift.

        # Arguments
            x: Input tensor. Must be 3D.
            intensity: Transformation intensity.
            channel_axis: Index of axis for channels in the input tensor.

        # Returns
            Numpy image tensor.

        """
        x = np.rollaxis(x, channel_axis, 0)
        min_x, max_x = np.min(x), np.max(x)
        channel_images = [
            np.clip(x_channel + intensity,
                    min_x,
                    max_x)
            for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    @staticmethod
    def transform_matrix_offset_center(matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    @staticmethod
    def flip_axis(x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x

    @staticmethod
    def apply_brightness_shift(x, brightness):
        """Performs a brightness shift.

        # Arguments
            x: Input tensor. Must be 3D.
            brightness: Float. The new brightness value.
            channel_axis: Index of axis for channels in the input tensor.

        # Returns
            Numpy image tensor.

        # Raises
            ValueError if `brightness_range` isn't a tuple.
        """
        if ImageEnhance is None:
            raise ImportError('Using brightness shifts requires PIL. '
                              'Install PIL or Pillow.')
        x = ImageTransformer.array_to_PIL_img(x)
        x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
        x = imgenhancer_Brightness.enhance(brightness)
        x = ImageTransformer.PIL_img_to_array(x)
        return x

    @staticmethod
    def array_to_PIL_img(x, data_format='channels_last', scale=True, dtype='float32'):
        """Converts a 3D Numpy array to a PIL Image instance.

        # Arguments
            x: Input Numpy array.
            data_format: Image data format, either "channels_first" or "channels_last".
                Default: "channels_last".
            scale: Whether to rescale the image such that minimum and maximum values
                are 0 and 255 respectively.
                Default: True.
            dtype: Dtype to use.
                Default: "float32".

        # Returns
            A PIL Image instance.

        # Raises
            ImportError: if PIL is not available.
            ValueError: if invalid `x` or `data_format` is passed.
        """
        if pil_image is None:
            raise ImportError('Could not import PIL.Image. '
                              'The use of `array_to_img` requires PIL.')
        x = np.asarray(x, dtype=dtype)
        if x.ndim != 3:
            raise ValueError('Expected image array to have rank 3 (single image). '
                             'Got array with shape: %s' % (x.shape,))

        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Invalid data_format: %s' % data_format)

        # Original Numpy array x has format (height, width, channel)
        # or (channel, height, width)
        # but target PIL image has format (width, height, channel)
        if data_format == 'channels_first':
            x = x.transpose(1, 2, 0)
        if scale:
            x = x - np.min(x)
            x_max = np.max(x)
            if x_max != 0:
                x /= x_max
            x *= 255

        if x.shape[2] == 1:
            return pil_image.fromarray(x[:, :, 0])
        else:
            raise ValueError('Unsupported channel number: %s' % (x.shape[2],))

    @staticmethod
    def PIL_img_to_array(img, data_format='channels_last', dtype='float32'):
        """Converts a PIL Image instance to a Numpy array.

        # Arguments
            img: PIL Image instance.
            data_format: Image data format,
                either "channels_first" or "channels_last".
            dtype: Dtype to use for the returned array.

        # Returns
            A 3D Numpy array.

        # Raises
            ValueError: if invalid `img` or `data_format` is passed.
        """
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: %s' % data_format)
        # Numpy array x has format (height, width, channel)
        # or (channel, height, width)
        # but original PIL image has format (width, height, channel)
        x = np.asarray(img, dtype=dtype)
        if len(x.shape) == 3:
            if data_format == 'channels_first':
                x = x.transpose(2, 0, 1)
        elif len(x.shape) == 2:
            if data_format == 'channels_first':
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: %s' % (x.shape,))
        return x
