import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tensorboard --logdir logs 查看图形

# 构造添加一个神经层的函数。
def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name='layer%s'%n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')

            # 显示
            tf.histogram_summary(layer_name+'/weights', Weights)   # tensorflow 0.12 以下版的
            # tf.summary.histogram(layer_name + '/weights', Weights) # tensorflow >= 0.12
        with tf.name_scope('Biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='B')

            tf.histogram_summary(layer_name+'/biase', biases)   # tensorflow 0.12 以下版的
            # tf.summary.histogram(layer_name + '/biases', biases)  # Tensorflow >= 0.12
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

            tf.histogram_summary(layer_name+'/outputs', outputs) # tensorflow 0.12 以下版本
            # tf.summary.histogram(layer_name + '/outputs', outputs) # Tensorflow >= 0.12
        return outputs

# 构建所需数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 利用占位符定义我们所需要的神经网络的输入
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 构建一个输入层1个、隐藏层10个、输出层1个的神经网络。
# 添加输入层
l1 = add_layer(xs, 1, 10, n_layer=1, activation_function=tf.nn.relu)
# 隐藏层
prediction = add_layer(l1, 10, 1, n_layer=2, activation_function=None)
# 计算误差
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
    tf.scalar_summary('loss',loss) # tensorflow < 0.12
    # tf.summary.scalar('loss', loss) # tensorflow >= 0.12
# 提升准确率
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 初始化
init = tf.initialize_all_variables() # tf 马上就要废弃这种写法
# init = tf.global_variables_initializer()  # 替换成这样就好

sess = tf.Session()


merged= tf.merge_all_summaries()    # tensorflow < 0.12
# merged = tf.summary.merge_all() # tensorflow >= 0.12

writer = tf.train.SummaryWriter('logs/', sess.graph)    # tensorflow < 0.12
# writer = tf.summary.FileWriter("logs/", sess.graph) # tensorflow >=0.12


sess.run(init)

# 绘制输入数据
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

# 开始学习
for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        rs = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(rs, i)
        # to see the step improvement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            passa
        # 绘制学习输出曲线
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.3)
