import os
import glob
import h5py
import numpy as np
import cv2

# read jpg image and store hdf5 each batch
def read_batch_images(files, img_size):
    num_X = len(files)
    X = np.zeros((num_X, img_size[0], img_size[1], 3),dtype=np.uint8)
    for i, f in enumerate(files):
        # read image
        x = cv2.imread(f)
        # BGR로 유지
        #x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        X[i] = cv2.resize(x, img_size, cv2.INTER_AREA)
        #X.append(cv2.resize(x, img_size, cv2.INTER_AREA))
    return X

# read whole images
def read_images(image_dir='temp/image/', file_name='temp/images.hdf5', batch_num=40000):
    # 파일 경로 지정
    image_path = image_dir + '*'

    # list images
    img_size = (224, 224)
    files = glob.glob(image_path)
    #################################
    num_files = len(files)
    # batch size(40000*224*224*3)
    batch_size = batch_num

    # check pre-loaded data
    # read jpg images
    if not os.path.isfile(file_name):
        with h5py.File(file_name, 'a') as f:
            # save indexes of items.csv
            index = []
            for file in files:
                index.append(os.path.basename(file)[:-8])
            index = np.sort(np.array(index, dtype=np.uint32))
            f.create_dataset('index', data=index, dtype=np.uint32)
            # save images
            f.create_dataset('image', (num_files, img_size[0], img_size[1], 3), dtype=np.uint8)
            cnt = 0
            for i in range(0, num_files, batch_size):
                f['image'][cnt * batch_size: (cnt + 1) * batch_size] = read_batch_images(files[i:i + batch_size], img_size)
                print(file_name, cnt)
                cnt += 1
    else:
        print('image hdf5 File Exists.')


if __name__=="__main__":
    read_images()
