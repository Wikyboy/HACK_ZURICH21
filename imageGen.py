import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from scipy.stats import truncnorm
import PIL.Image
import numpy as np
import IPython.display
import io

tf.disable_v2_behavior()
module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'
tf.reset_default_graph()
module = hub.Module(module_path)
inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
            for k, v in module.get_input_info_dict().items()}
output = module(inputs)

input_z = inputs['z']
input_y = inputs['y']
input_trunc = inputs['truncation']

dim_z = input_z.shape.as_list()[1]
vocab_size = input_y.shape.as_list()[1]

initializer = tf.global_variables_initializer()
sess = tf.Session()
sess.run(initializer)


class ImageGen():

    def truncated_z_sample(self, batch_size, truncation=1., seed=None):
        state = None if seed is None else np.random.RandomState(seed)
        values = truncnorm.rvs(-2, 2, size=(batch_size,
                                            dim_z), random_state=state)
        return truncation * values

    def one_hot(self, index):
        index = np.asarray(index)
        if len(index.shape) == 0:
            index = np.asarray([index])
        assert len(index.shape) == 1
        num = index.shape[0]
        output = np.zeros((num, vocab_size), dtype=np.float32)
        output[np.arange(num), index] = 1
        return output

    def one_hot_if_needed(self, label):
        label = np.asarray(label)
        if len(label.shape) <= 1:
            label = self.one_hot(label)
        assert len(label.shape) == 2
        return label

    def sample(self, sess, noise, label, truncation=1., batch_size=8):
        noise = np.asarray(noise)
        label = np.asarray(label)
        num = noise.shape[0]
        if len(label.shape) == 0:
            label = np.asarray([label] * num)
        if label.shape[0] != num:
            raise ValueError(
                'Got # noise samples ({}) != # label samples ({})'.format(
                    noise.shape[0], label.shape[0]))
        label = self.one_hot_if_needed(label)
        ims = []
        for batch_start in range(0, num, batch_size):
            s = slice(batch_start, min(num, batch_start + batch_size))
            feed_dict = {
                input_z: noise[s],
                input_y: label[s],
                input_trunc: truncation
            }
            ims.append(sess.run(output, feed_dict=feed_dict))
        ims = np.concatenate(ims, axis=0)
        assert ims.shape[0] == num
        ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
        ims = np.uint8(ims)
        return ims

    def midpoint(self, A, B):
        return np.array([0.5*(A+B)])

    def interpolate(self, A, B, num_interps):
        if A.shape != B.shape:
            raise ValueError(
                'A and B must have the same shape to interpolate.')
        alphas = np.linspace(0, 1, num_interps)
        return np.array([(1-a)*A + a*B for a in alphas])

    def imgrid(self, imarray, cols=5, pad=1):
        if imarray.dtype != np.uint8:
            raise ValueError('imgrid input imarray must be uint8')
        pad = int(pad)
        assert pad >= 0
        cols = int(cols)
        assert cols >= 1
        N, H, W, C = imarray.shape
        rows = N // cols + int(N % cols != 0)
        batch_pad = rows * cols - N
        assert batch_pad >= 0
        post_pad = [batch_pad, pad, pad, 0]
        pad_arg = [[0, p] for p in post_pad]
        imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
        H += pad
        W += pad
        grid = (imarray
                .reshape(rows, cols, H, W, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(rows*H, cols*W, C))
        if pad:
            grid = grid[:-pad, :-pad]
        return grid

    def imshow(self, a, format='png', jpeg_fallback=True):
        a = np.asarray(a, dtype=np.uint8)
        data = io.BytesIO()
        PIL.Image.fromarray(a).save(data, format)
        im_data = data.getvalue()
        try:
            disp = IPython.display.display(IPython.display.Image(im_data))
        except IOError:
            if jpeg_fallback and format != 'jpeg':
                print(('Warning: image was too large to display in format "{}"; '
                       'trying jpeg instead.').format(format))
                return self.imshow(a, format='jpeg')
            else:
                raise
        return disp

    def interpolate_and_shape(self, A, B, num_interps, num_samples):
        interps = self.midpoint(A, B)
        return (interps.transpose(1, 0, *range(2, len(interps.shape))).reshape(
            num_samples * num_interps, *interps.shape[2:]))

    def generate_image(self, categories: list):
        category_A = categories[0]
        num_samples = 1
        num_interps = 1
        truncation = 0.2
        noise_seed_A = 0
        noise_seed_B = 0
        category_B = categories[1]

        z_A, z_B = [self.truncated_z_sample(
            num_samples, truncation, noise_seed)
                    for noise_seed in [noise_seed_A, noise_seed_B]]
        y_A, y_B = [self.one_hot(
            [int(category.split(')')[0])] * num_samples)
                    for category in [category_A, category_B]]

        z_interp = self.interpolate_and_shape(z_A, z_B, num_interps, num_samples)
        y_interp = self.interpolate_and_shape(y_A, y_B, num_interps, num_samples)

        ims = self.sample(sess, z_interp, y_interp, truncation=truncation)
        self.imshow(self.imgrid(ims, cols=num_interps))
        plt.imsave('creature.jpeg', ims[0])
