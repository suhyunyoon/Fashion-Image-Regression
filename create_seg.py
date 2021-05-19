import numpy as np
import h5py

# Create Semantic segmentation with IRN
def gen_mask(image_file='temp/images.hdf5', mask_file='temp/masks.hdf5', batch_size=40000, img_size=(224, 224)):
    with h5py.File(image_file, 'r') as f, h5py.File(mask_file, 'a') as m:
        num_files = len(f['image'])
        try:
            m.create_dataset('mask', (num_files, img_size[0], img_size[1], 1), dtype=np.bool)
        except:
            del m['mask']
            m.create_dataset('mask', (num_files, img_size[0], img_size[1], 1), dtype=np.bool)
        cnt = 0
        for index in range(0, num_files, batch_size):
            num_X = len(f['index'])
            mask = np.zeros((num_X, img_size[0], img_size[1], 1), dtype=np.bool)
            # 일단 아무거나 생성
            pivot = img_size[0] // 2
            mask[:,pivot-30:pivot+30, pivot-30:pivot+30, :] = True

            # save masks
            m['mask'][cnt * batch_size: (cnt + 1) * batch_size] = mask

            cnt += 1
            print('\nmasks {} Done.'.format(cnt))

    print('Done')