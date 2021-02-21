import os
import argparse
import cv2
import collections
import numpy as np

from models.dense_net import DenseNet

img_size = 64

# # FER+ MODEL
# images_mean = 129
# images_std = 63.58

# FER+-MEC finetune MODEL
images_mean = 106
images_std = 58

def extract_feature_batch():
  args = parse_opts()
  model_params = vars(args)
  batch_size = args.batch_size

  print("Initialize the model..")
  # fake data_provider
  DataProvider = collections.namedtuple('DataProvider', ['data_shape', 'n_classes'])
  data_provider = DataProvider(data_shape=(img_size, img_size, 1), n_classes=10)
  model = DenseNet(data_provider=data_provider, **model_params)
  model.saver.restore(model.sess, args.model_path)
  print("Successfully load model from model path: %s" % args.model_path)

  video_names = [x for x in os.listdir(args.face_dir)]
  video_names.sort()
  avg_num_imgs = 0

  for vid, video_name in enumerate(video_names):
    video_dir = os.path.join(args.face_dir, video_name)
    img_paths = os.listdir(video_dir)
    if len(img_paths) == 0:
      continue

    output_subdir = os.path.join(args.outft_dir, video_name)
    if os.path.exists(output_subdir):
      continue
    else:
      os.makedirs(output_subdir)

    img_paths.sort(key=lambda x:int(x.split('.')[0]))
    avg_num_imgs += len(img_paths)

    imgs = []
    for img_path in img_paths:
      img_path = os.path.join(video_dir, img_path)
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      img = cv2.resize(img, (img_size, img_size))
      imgs.append(img)

    imgs = (np.array(imgs, np.float32) - images_mean) / images_std
    imgs = np.expand_dims(imgs, 3)
    # pool4.shape=(batch_size, 4, 4, 256)
    # fc5.shape=fc6.shape=(batch_size, 1, 1, 512)
    # prob.shape=(batch_size, num_classes)
    fcs, probs = [], []
    for i in xrange(0, imgs.shape[0], batch_size):
      feed_dict = {
        model.images: imgs[i: i + batch_size],
        model.is_training: False
      }
      fc, prob = model.sess.run(
        [end_points['fc'], end_points['preds']],
        feed_dict=feed_dict)
      # prev_last_pools.extend(prev_last_pool)
      fcs.extend(fc)
      probs.extend(prob)

    # prev_last_pools = np.array(prev_last_pools, np.float32)
    fcs = np.array(fcs, np.float32)
    probs = np.array(probs, np.float32)

    
    # with open(os.path.join(output_subdir, 'pool.npy'), 'wb') as f:
    #   np.save(f, prev_last_pools)
    with open(os.path.join(output_subdir, 'fc.npy'), 'wb') as f:
      np.save(f, fcs)
    with open(os.path.join(output_subdir, 'prob.npy'), 'wb') as f:
      np.save(f, probs)

    print(vid, video_name, len(img_paths), 
      fcs.shape, probs.shape)

  avg_num_imgs /= float(len(video_names))
  print('average faces per video', avg_num_imgs)
  

if __name__ == '__main__':
  extract_feature_batch()