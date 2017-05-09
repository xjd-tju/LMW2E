import rnn_utils as input_reader
import RNN as rnn
import data_utils as utils
import tensorflow as tf

# with tf.Graph().as_default(),tf.Session() as session:

    # test conv and pool
    # input = tf.reshape(tf.range(1,51,dtype=tf.float32),[1,5,10,1])
    # init = tf.reshape(tf.range(1,21,dtype=tf.float32),[2,10,1,1])
    # kerenl = tf.Variable(init)
    # conv = tf.nn.conv2d(input,kerenl,strides=[1, 1, 1, 1], padding='VALID')
    # # pool = tf.nn.max_pool(conv, [1, 4, 1, 1], [1, 1, 1, 1], 'VALID')
    # pool = tf.reduce_max(conv)
    # initializer = tf.global_variables_initializer()
    # session.run(initializer)
    # print(session.run(conv))
    # print(session.run(pool))
    # print(pool)
    # print(pool.get_shape().as_list())

    # test contact

    # a = [1,2,3,4]
    # b = [4,5,6,7]
    # c = tf.concat([a,b],axis=0)
    # print(session.run(c))

    # TEST  char vocab
    # cv = utils.pkl_load(utils.CHAR_VOCAB_NAME)
    # for k in cv.keys():
    #     print(k,cv[k])

#     Test placeholder
import numpy as np
x_data = np.array( [[1,2],[4,5,6],[1,2,3,4,5,6]] )
lens = np.array([len(x_data[i]) for i in range(len(x_data))])
mask = np.arange(lens.max()) < lens[:,None]
padded = np.zeros(mask.shape)
padded[mask] = np.hstack((x_data[:]))
# print(a)
with tf.Graph().as_default(),tf.Session() as session:
    tf_pl = tf.placeholder(dtype=tf.int32,shape=[3,None])
    tf_mask = tf.placeholder(dtype=tf.bool,shape=[3,None])
    for i in range(3):
        masked = tf.boolean_mask(tf_pl[i],tf_mask[i])
    re = masked - 1
print(session.run(re,feed_dict={tf_pl:padded,tf_mask:mask}))
