import tensorflow as tf


class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()
# tf v 1.x:
x_input = tf.constant(3.0)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(model.W))
    print(sess.run(model(x_input)))


# TF v 2.0:
#model(3.0).numpy()