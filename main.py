import numpy as np
"""
2021. 09. 04.
edited by LCS
"""

"""
make patches
"""
def get_patches(img_arr, size=128, stride=128):
    patched_list = []
    overlapping = 0
    if stride != size:
        overlapping = (size // stride) - 1
    if img_arr.ndim == 3:
        i_max = img_arr.shape[0] // stride - overlapping
        for i in range(i_max):
            for j in range(i_max):
                for k in range(i_max):
                    patched_list.append(img_arr[i * stride: i * stride + size, j * stride: j * stride + size,
                                        k * stride: k * stride + size, ])
    else:
        raise ValueError("img_arr.ndim must be equal 4")
    return np.stack(patched_list)

"""
reconstruct patches data
"""
def reconstruct_patch(img_arr, org_img_size, stride=128, size=128):
    if type(org_img_size) is not tuple:
        raise ValueError("org_image_size must be a tuple")
    if size is None:
        size = img_arr.shape[2]
    if stride is None:
        stride = size
    nm_layers = img_arr.shape[4]
    i_max = (org_img_size[0] // stride ) + 1 - (size // stride)
    j_max = (org_img_size[1] // stride ) + 1 - (size // stride)
    k_max = (org_img_size[2] // stride ) + 1 - (size // stride)
    total_nm_images = img_arr.shape[0] // (i_max ** 3)
    images_list = []
    kk=0
    for img_count in range(total_nm_images):
        img_bg = np.zeros((org_img_size[0],org_img_size[1],org_img_size[2],nm_layers), dtype=img_arr[0].dtype)
        for i in range(i_max):
            for j in range(j_max):
                for k in range(k_max):
                    for layer in range(nm_layers):
                        img_bg[
                        i * stride: i * stride + size,
                        j * stride: j * stride + size,
                        k * stride: k * stride + size,
                        layer,
                        ] = img_arr[kk, :, :, :, layer]
                    kk += 1
        images_list.append(img_bg)
    return np.stack(images_list)




"""
example: 1 big cube (256,256,256)
"""
big_cube = np.random.rand(256,256,256)
print('1 big cube shape : ', big_cube.shape)
patched_cube = get_patches(img_arr=big_cube, size=128, stride=128)
print('patched data shape : ', patched_cube.shape)
reconstructed = np.squeeze(reconstruct_patch(img_arr=np.expand_dims(patched_cube,axis=-1),
                                             org_img_size=(256,256,256), stride=128))
print('reconstructed data shape', reconstructed.shape)
print('--------------------------------------')

"""
example: 4 big cubes (4, 256, 256, 256)
"""
def get_patches_data(data):
    patches = []
    for i in range(data.shape[0]):
        print('i th : ',i)
        print(data[i].shape)
        patched_cube = get_patches(img_arr=data[i], size=128, stride=128)
        patches.append(patched_cube)
    patches = np.vstack(patches)
    print('patched cube shape : ', patches.shape)
    return patches
print('-------------- get_patch --------------')
big_cubes = np.random.rand(4,256,256,256)
print('4 big cubes shape : ', big_cubes.shape)
get_patches_data(big_cubes)
print('-------------- done --------------')

def reconstructed_patches_data(data):
    patches = []
    for i in range(int(data.shape[0]/8)):
        print('i th : ', i)
        print(data[i].shape)
        print(data[8 * i : 8 * i + 8].shape)
        reconstructed = np.squeeze(reconstruct_patch(img_arr=np.expand_dims(data[8 * i : 8 * i + 8],axis=-1),                                                   org_img_size=(256,256,256), stride=128))
        print('reconstructed data shape', reconstructed.shape)
        patches.append(np.expand_dims(reconstructed,axis=0))
    patches = np.vstack(patches)
    print('All reconstructed data shape : ', patches.shape)
    return patches
print('-------------- reconstruct_patch --------------')
big_cubes = np.random.rand(4,256,256,256)
print('4 big cubes shape : ', big_cubes.shape)
reconstructed_patches_data(get_patches_data(big_cubes))
print('-------------- done --------------')