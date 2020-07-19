import numpy as np

def disp_pyr_vertical(pyr):
    """
    Visualize the pyramid vertically
    """
    num_levels = len(pyr)

    H, W = pyr[0].shape

    img_heights = [H * 2 ** (-i) for i in np.arange(num_levels, dtype=float)]
    H = np.int(np.sum(img_heights))

    out = np.zeros((H, W))

    for i in np.arange(num_levels):
        rstart = np.int(np.sum(img_heights[:i]))
        rend = np.int(rstart + img_heights[i])

        out[rstart:rend, :np.int(img_heights[i])] = pyr[i]

    return out


def disp_pyr_horizontal(pyr):
    """
    Visualize the pyramid horizontally
    """
    num_levels = len(pyr)

    H, W = pyr[0].shape

    img_width = [W * 2 ** (-i) for i in np.arange(num_levels, dtype=float)]
    # print(img_width) # [512.0, 256.0, 128.0, 64.0]
    W = np.int(np.sum(img_width))
    # print(W) # 960
    out = np.zeros((H, W))

    for i in np.arange(num_levels):
        rstart = np.int(np.sum(img_width[:i]))
        rend = np.int(rstart + img_width[i])
        print(rstart)
        print(rend)
        print()
        # out[rstart:rend, :np.int(img_width[i])] = pyr[i]
        out[:np.int(img_width[i]), rstart:rend] = pyr[i]

    return out
