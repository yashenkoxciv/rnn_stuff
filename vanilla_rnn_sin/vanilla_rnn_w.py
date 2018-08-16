import numpy as np
import tensorflow as tf
from sindata import sin_data, next_batch, next_unrolling

UNROLLING = 3
X_DIM = 1
Y_DIM = 1
H_DIM = 50

def unroll(h_p, x):
    with tf.variable_scope('rnn_block', reuse=tf.AUTO_REUSE):
        w_h = tf.get_variable('w_h', [H_DIM, H_DIM])
        w_x = tf.get_variable('w_x', [X_DIM, H_DIM])
        b = tf.get_variable('b', [H_DIM])
        h = tf.nn.tanh(
                tf.matmul(tf.reshape(h_p, [1, H_DIM]), w_h) +\
                tf.matmul(tf.reshape(x, [1, X_DIM]), w_x) + b
        )
    return tf.reshape(h, [H_DIM])

def rnn_unrolling(x):
    return tf.scan(unroll, x, initializer=tf.zeros([H_DIM]))

def forward_propagation(s):
    with tf.variable_scope('dense1', reuse=tf.AUTO_REUSE):
        w = tf.get_variable('w', [H_DIM, Y_DIM])
        b = tf.get_variable('b', [Y_DIM])
        p = tf.matmul(s, w) + b # tf.nn.sigmoid
    return p

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None], 'x')
    y = tf.placeholder(tf.float32, [None], 'y')
    
    s = rnn_unrolling(x) # rnn_states
    p = forward_propagation(s[-1:])
    
    loss = tf.reduce_mean(tf.square(y - p))
    #opt = tf.train.AdamOptimizer().minimize(loss)
    opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
    
    # summary
    tf.summary.scalar('loss', loss)
    all_summary = tf.summary.merge_all()
    
    # start session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('summary/', sess.graph)
    saver = tf.train.Saver()
    
    # load toy data
    _, xs = sin_data(50)
    
    # set train parameters
    epochs = 1000
    batch_size = 10
    batches = xs.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, y_batch = next_unrolling(xs, UNROLLING)
            
            sess.run(opt, {x: x_batch, y: y_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100), end='', flush=True
            )
            
            # write summary
            summary_str = sess.run(all_summary, {
                x: x_batch,
                y: y_batch
            })
            writer.add_summary(summary_str, epoch)
    print('\rDone', ' '*25, flush=True)
    
    # closing
    saver.save(sess, 'model/model.ckpt')
    writer.close()
    sess.close()
    
    
    

