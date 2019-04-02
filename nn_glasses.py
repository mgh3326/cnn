import tensorflow as tf

def crop_image(images):
    shape = tf.shape(images)
    height = shape[0]
    width = shape[1]

    offset_y = height / 4

    images = tf.image.crop_to_bounding_box(images, offset_y, 0, offset_y*2, width)
    return tf.image.resize_image_with_crop_or_pad(images, height, width)



def inference(images, keep_prob, batch_size, image_crop_size, input_channels, num_of_classes, variable_with_weight_decay, variable_on_cpu, activation_summary, log_input, log_feature):
    batch_size = tf.cast(batch_size, tf.int32)
    images = tf.reshape(images, [batch_size, image_crop_size, image_crop_size, input_channels])

    if log_input:
        width = images[0].get_shape()[0].value
        height = images[0].get_shape()[1].value
        img = tf.reshape(images[0], [1, width, height, input_channels])
        tf.summary.image('image', img, 1)


    with tf.variable_scope('conv1_1') as scope:
        out_channels = 32
        kernel = variable_with_weight_decay('weights', shape=[3,3,input_channels,out_channels], stddev=0.01)
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv
    #pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
    #pool_layer = tf.nn.lrn(pool_layer, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')


    with tf.variable_scope('conv1_2') as scope:
        out_channels = 32
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3], out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        #biases = variable_on_cpu('biases', [out_channels2], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    #conv = tf.nn.lrn(conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')


    with tf.variable_scope('conv2_1') as scope:
        out_channels = 64
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv
    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    #pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')

    with tf.variable_scope('conv2_2') as scope:
        out_channels = 64
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv

    with tf.variable_scope('conv2_3') as scope:
        out_channels = 64
        kernel = variable_with_weight_decay('weights', shape=[7,7,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')


    with tf.variable_scope('fc3') as scope:
        size = pool_layer.get_shape()[1] * pool_layer.get_shape()[2] * pool_layer.get_shape()[3]
        reshape = tf.reshape(pool_layer, tf.stack([-1, size]))
        weights = variable_with_weight_decay('weights', shape=[size, 256], stddev=0.05)
        biases = variable_on_cpu('biases', [256], tf.contrib.keras.initializers.he_normal())
        local_layer = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        activation_summary(local_layer)

    #keep_prob1 = tf.convert_to_tensor(0.7)
    local_dropout = tf.nn.dropout(local_layer, keep_prob)


    with tf.variable_scope('fc4') as scope:
        weights = variable_with_weight_decay('weights', shape=[256, 256], stddev=0.04)
        #biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [256], tf.contrib.keras.initializers.he_normal())
        local_dropout = tf.nn.relu(tf.matmul(local_dropout, weights) + biases, name=scope.name)
        activation_summary(local_dropout)

    local_dropout = tf.nn.dropout(local_dropout, keep_prob)


    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('wegiths', [256, num_of_classes], stddev=0.04)
        #biases = variable_on_cpu('biases', [num_of_classes], tf.constant_initializer(0.1))
        biases = variable_on_cpu('biases', [num_of_classes], tf.contrib.keras.initializers.he_normal())
        softmax_linear = tf.add(tf.matmul(local_dropout, weights), biases, name=scope.name)
        activation_summary(softmax_linear)
        softmax = tf.nn.softmax(softmax_linear, name="softmax")

    return softmax_linear