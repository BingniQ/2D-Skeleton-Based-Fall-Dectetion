
import numpy as np
import pandas as pd
import os
import csv
import tensorflow as tf
import PLSTM.PLSTM, PLSTM.Dataloader
from PLSTM.Features_PLSTM import batch_size, lr, n_layers, n_frame, step

data_path = "../dataset/Fall2_Cam5.avi_keys.csv"
model_path = "../model"
preds_path = "dataset/preds.csv"
action_names = ["non-fall", "Falling"]


def eval_f(sess, model, data):
    feed_dict = {model.skel_input: data, model.plstm_keep_prob: 1.00}

    return sess.run(model.action_distribution, feed_dict)


a = 0
b = n_frame

while os.path.getsize(data_path) == 0 or pd.read_csv(data_path, header=0, sep=';').shape[0] < b:
    pass

with open("../dataset/Fall2_Cam5.avi_keys.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')  # change contents to floats
    # get header from first row
    header = next(reader)
    # get all the rows as a list
    source_data = list(reader)
    # transform data into numpy array
    source_data = np.array(source_data).astype(float)

skeleton_data=source_data[:,0:36]
dat = np.reshape(skeleton_data, (-1, 18, 2))
print(dat.shape)
dl = PLSTM.Dataloader.DataLoader(skel_train=dat, skel_test=None, mode="train")

# config = tf.ConfigProto()
# sess = tf.Session(config=config)
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.restore(sess, model_path)
#
data = dl.get_samples(batch_size, 0, is_rotate=True)
# preds = eval_f(sess, model, data)
# df = pd.DataFrame(data=preds, columns=action_names)
# df.to_csv(preds_path, sep=';', index=False)
#
# while True:
#     a += step
#     b += step
#
#     while pd.read_csv(data_path, header=None, sep=';').shape[0] < b:
#         pass
#
#     dat = np.array([pd.read_csv(data_path, header=None, sep=';', skiprows=a, nrows=b).values])
#     dl = dataloader_PLSTM.DataLoader(skel_train=None, skel_test=dat, mode="test")
#
#     data = dl.get_test_sample(batch_size, 0, is_rotate=True)
#     preds = eval_f(sess, model, data)
#     df = df.append(pd.DataFrame(data=preds, columns=action_names))
#     df.to_csv(preds_path, sep=';', index=False)
