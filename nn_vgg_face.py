import tensorflow as tf


def crop_image(images):
    return images


def inference(images, keep_prob, batch_size, image_crop_size, input_channels, num_of_classes, variable_with_weight_decay, variable_on_cpu, activation_summary, log_input, log_feature):
    batch_size = tf.cast(batch_size, tf.int32)
    images = tf.reshape(images, [batch_size, image_crop_size, image_crop_size, input_channels])

    if log_input:
        width = images[0].get_shape()[0].value
        height = images[0].get_shape()[1].value
        img = tf.reshape(images[0], [1, width, height, input_channels])
        tf.summary.image('image', img, 1)


    with tf.variable_scope('conv1_1') as scope:
        out_channels = 64
        kernel = variable_with_weight_decay('weights', shape=[3,3,input_channels,out_channels], stddev=0.01)
        conv = tf.nn.conv2d(images, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu1_1")
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
        out_channels = 64
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3], out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu1_2")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')


    with tf.variable_scope('conv2_1') as scope:
        out_channels = 128
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu2_1")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    #norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    #pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
    pool_layer = conv


    with tf.variable_scope('conv2_2') as scope:
        out_channels = 128
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu2_2")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')


    with tf.variable_scope('conv3_1') as scope:
        out_channels = 256
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu3_1")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv


    with tf.variable_scope('conv3_2') as scope:
        out_channels = 256
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu3_2")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv


    with tf.variable_scope('conv3_3') as scope:
        out_channels = 256
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu3_3")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')


    with tf.variable_scope('conv4_1') as scope:
        out_channels = 512
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu4_1")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv

    with tf.variable_scope('conv4_2') as scope:
        out_channels = 512
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu4_2")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv

    with tf.variable_scope('conv4_3') as scope:
        out_channels = 512
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu4_3")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')


    with tf.variable_scope('conv5_1') as scope:
        out_channels = 512
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu5_1")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv

    with tf.variable_scope('conv5_2') as scope:
        out_channels = 512
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu5_2")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = conv

    with tf.variable_scope('conv5_3') as scope:
        out_channels = 512
        kernel = variable_with_weight_decay('weights', shape=[3,3,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='SAME')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu5_3")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool5')

    with tf.variable_scope('fc6') as scope:
        out_channels = 4096
        kernel = variable_with_weight_decay('weights', shape=[7,7,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='VALID')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu6")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.dropout(conv, keep_prob)

    with tf.variable_scope('fc7') as scope:
        out_channels = 4096
        kernel = variable_with_weight_decay('weights', shape=[1,1,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='VALID')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(pre_activation, name="relu7")
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)

    pool_layer = tf.nn.dropout(conv, keep_prob)

    with tf.variable_scope('fc8') as scope:
        out_channels = num_of_classes
        kernel = variable_with_weight_decay('weights', shape=[1,1,pool_layer.get_shape()[3],out_channels], stddev=0.05)
        conv = tf.nn.conv2d(pool_layer, kernel, [1,1,1,1], padding='VALID')
        biases = variable_on_cpu('biases', [out_channels], tf.contrib.keras.initializers.he_normal())
        pre_activation = tf.nn.bias_add(conv, biases)
        conv = pre_activation
        activation_summary(conv)

        if log_feature:
            width = conv[0].get_shape()[0].value
            height = conv[0].get_shape()[1].value
            imgs = tf.reshape(conv[0], [1,width,height,out_channels])
            imgs = tf.transpose(imgs, [3,1,2,0])
            tf.summary.image(conv.op.name, imgs, 10)
    
    pool_layer = conv

    with tf.variable_scope('softmax_linear') as scope:
        softmax_linear = tf.reshape(pool_layer, [-1, num_of_classes])
        activation_summary(softmax_linear)
        softmax = tf.nn.softmax(softmax_linear, name="softmax")

    return softmax_linear