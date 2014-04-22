import matplotlib.pyplot as plt
import numpy as np

def plot_images(imgs, loc, title=None):
    '''Plot an array of images.

    We assume that we are given a matrix of data whose shape is (n*n, s*s) --
    that is, there are n^2 images along the first axis of the array, and each
    image is a square measuring s pixels on a side. Each row of the input will
    be plotted as a sub-region within a single image array containing an n x n
    grid of images.
    '''
    n = int(np.sqrt(len(imgs)))
    assert n * n == len(imgs), 'images array must contain a square number of rows!'
    s = int(np.sqrt(len(imgs[0])))
    assert s * s == len(imgs[0]), 'images must be square!'

    img = np.zeros((s * n, s * n), dtype=imgs[0].dtype)
    for i, pix in enumerate(imgs):
        r, c = divmod(i, n)
        img[r * s:(r+1) * s, c * s:(c+1) * s] = pix.reshape((s, s))

    ax = plt.gcf().add_subplot(loc)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.imshow(img, cmap=plt.cm.gray)
    if title:
        ax.set_title(title)

def plot_layers(weights, tied_weights=False):
    '''Create a plot of weights, visualized as "bottom-level" pixel arrays.'''
    if hasattr(weights[0], 'eval'):
        weights = [w.eval() for w in weights]
    k = min(len(weights), 9)
    imgs = np.eye(weights[0].shape[0])
    for i, weight in enumerate(weights[:-1]):
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs, 100 + 10 * k + i + 1, 'Layer {}'.format(i+1))
    weight = weights[-1]
    if int(np.sqrt(weight.shape[1])) ** 2 != weight.shape[1]:
        return
    if tied_weights:
        imgs = np.dot(weight.T, imgs)
        plot_images(imgs, 100 + 10 * k + k, 'Layer {}'.format(k))
    else:
        plot_images(weight, 100 + 10 * k + k, 'Decoding weights')

def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar

def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`PIL.Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                        in zip(img_shape, tile_shape, tile_spacing)]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output np ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                    dtype=X.dtype)

        #colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in xrange(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(out_shape,
                        dtype=dt) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in xrange(tile_shape[0]):
            for tile_col in xrange(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                        tile_row * (H + Hs): tile_row * (H + Hs) + H,
                        tile_col * (W + Ws): tile_col * (W + Ws) + W
                        ] = this_img * c
        return out_array
