import numpy as np
import tensorflow as tf
from toydata import X_DIM, POS, next_batch

TRAIN_SIZE = 60
TEST_SIZE = 100

def atn(x):
    if hasattr(atn, 'reuse'):
        atn.reuse = True
    else:
        atn.reuse = False
    m = tf.layers.dense(
            x, X_DIM,
            activation=tf.nn.softmax, name='n_atn', reuse=atn.reuse
    )
    return x*m

def n(x):
    if hasattr(n, 'reuse'):
        n.reuse = True
    else:
        n.reuse = False
    atn_x = atn(x)
    logits = tf.layers.dense(atn_x, 1, name='n_logits', reuse=n.reuse)
    return logits

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, X_DIM], 'x')
    y = tf.placeholder(tf.float32, [None, 1], 'y')
    
    x_logits = n(x)
    
    loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x_logits)
    )
    opt = tf.train.AdamOptimizer().minimize(loss)
    
    '''
    acc = tf.reduce_mean(
            tf.cast(
                    tf.equal(y, tf.round(tf.nn.sigmoid(x_logits))),
                    tf.float32
            ))
    '''
    delta = tf.abs(y - tf.nn.sigmoid(x_logits))
    correct_preds = tf.cast(tf.less(delta, 0.5), tf.int32)
    acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    
    # summary
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    all_summary = tf.summary.merge_all()
    
    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('atn_summary/', sess.graph)
    saver = tf.train.Saver()
    
    # set train parameters
    epochs = 100
    batch_size = 100
    batches = 100
    batch_end = epochs*batches
    batch_step = 0
    
    test_x, test_y = next_batch(5000, X_DIM, POS)
    train_x, train_y = next_batch(TRAIN_SIZE, X_DIM, POS)
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            batch_idx = np.random.choice(TRAIN_SIZE, batch_size)
            x_batch, y_batch = train_x[batch_idx], train_y[batch_idx]
            
            sess.run(opt, {x: x_batch, y: y_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
        # write summary
        summary_str = sess.run(all_summary, {
            x: test_x,
            y: test_y
        })
        writer.add_summary(summary_str, epoch-1)
    print('\rDone', ' '*25, flush=True)
    
    # closing
    saver.save(sess, 'atn_model/model.ckpt')
    writer.close()
    sess.close()
