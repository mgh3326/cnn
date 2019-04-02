from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from cnn import CNN

import os
import re
import time
import numpy as np
from datetime import datetime
import math
import json
from shutil import copyfile
from PIL import Image
import pickle
import operator

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'eval',
                           '''Mode {train | eval | export | single | resize | pb | map-export | map-inference} DEFAULT=eval''')
tf.app.flags.DEFINE_string('data_dir', None, '''Path to data.''')
tf.app.flags.DEFINE_string('ckpt', None, '''The directory where checkpoint file is.''')
tf.app.flags.DEFINE_string('network', None, '''File Path to network.''')
tf.app.flags.DEFINE_string('pb', None, '''pb File.''')
tf.app.flags.DEFINE_string('map', None, '''Map File Path.''')
tf.app.flags.DEFINE_string('out_node', "softmax_linear/softmax_linear:0",
                           '''Name of out node. DEFAULT=softmax_linear/softmax''')

tf.app.flags.DEFINE_string('event_dir', None, '''Path to events.''')
tf.app.flags.DEFINE_string('out_dir', None, '''Path to data output(for resize mode).''')
tf.app.flags.DEFINE_integer('model_version', 1, '''Model version for Tensorflow Serving.''')

tf.app.flags.DEFINE_string('image', None, '''Image file for single evaluation.''')
tf.app.flags.DEFINE_integer('num_of_classes', None, '''Number of classes.''')

tf.app.flags.DEFINE_string('title', 'unknown', '''Name of network. DEFAULT=unknown''')
tf.app.flags.DEFINE_boolean('scaling', True, '''Scaling Input Images. DEFAULT=True''')
tf.app.flags.DEFINE_integer('image_size', 100, '''Image size(width == height) DEFAULT=100''')
tf.app.flags.DEFINE_integer('image_crop_size', 80, '''Image crop size(width == height) DEFAULT=80''')
tf.app.flags.DEFINE_boolean('log_input', True, '''Log input image. DEFAULT=True''')
tf.app.flags.DEFINE_boolean('grayscale', True, '''Make input images to grayscale. DEFAULT=True''')
tf.app.flags.DEFINE_boolean('log_feature', False, '''Log feature maps. DEFAULT=False''')
tf.app.flags.DEFINE_boolean('use_fp16', False, '''Use float16. DEFAULT=False''')
tf.app.flags.DEFINE_integer('max_steps', 20000, '''Max steps for training. DEFAULT=20000''')
tf.app.flags.DEFINE_integer('save_steps', 100, '''How often to save steps. DEFAULT=100''')
tf.app.flags.DEFINE_integer('batch_size', 64, '''Max steps for training. DEFAULT=64''')
tf.app.flags.DEFINE_boolean('continue_train', True, """Training continued. DEFAULT=True""")
tf.app.flags.DEFINE_integer('log_frequency', 100, """How often to log result to the console. DEFAULT=100""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Where to log device placement. DEFAULT=False""")
tf.app.flags.DEFINE_string('error_dir', None, '''Path to copy errors.''')
tf.app.flags.DEFINE_float('learning_rate', 0.005, '''Learning rate. DEFAULT = 0.005''')

IMAGE_FILE_EXTS = ['.jpg', '.png', '.jpeg']


def _num_of_folders(dir):
    return len([d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))])


def _num_of_files(root_dir):
    return sum([len(files) for r, d, files in os.walk(root_dir)])


def resize():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')


def train():
    # PreProcess
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    num_of_classes = _num_of_folders(FLAGS.data_dir)
    if num_of_classes <= 0:
        raise ValueError('Invalid num_of_classes')

    num_of_samples = _num_of_files(FLAGS.data_dir)
    if num_of_samples == None:
        raise ValueError('Please supply num_of_samples.')

    print('[ SUMMARY ]')
    print('Num of classes: {}'.format(num_of_classes))
    print('Num of samples: {}'.format(num_of_samples))

    # Training
    with tf.Graph().as_default():
        info = [
            ['Title', FLAGS.title],
            ['Number of classes', str(num_of_classes)],
            ['Number of samples', str(num_of_samples)],
            ['Image size', str(FLAGS.image_size)],
            ['Image Crop size', str(FLAGS.image_crop_size)],
            ['Grayscale', str(FLAGS.grayscale)],
            ['Use Float16', str(FLAGS.use_fp16)]
        ]
        tf.summary.text('NetworkInfo', tf.convert_to_tensor(info), collections=[])

        nn = CNN(FLAGS.network, num_of_classes, num_of_samples, FLAGS.image_size, FLAGS.image_crop_size,
                 FLAGS.log_input, FLAGS.grayscale, FLAGS.log_feature, FLAGS.use_fp16)
        global_step = tf.contrib.framework.get_or_create_global_step()

        with tf.device('/cpu:0'):
            images, labels, filenames = nn.inputs(FLAGS.data_dir, FLAGS.batch_size, FLAGS.batch_size * 100,
                                                  FLAGS.scaling, True)

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        keep_prob = tf.placeholder(dtype, name="keep_prob")
        batch_size = tf.placeholder(tf.float32, name="batch_size")
        logits = nn.inference(images, keep_prob, batch_size)
        loss, accuracy = nn.loss(logits, labels)

        train_op = nn.train(loss, global_step, FLAGS.batch_size, FLAGS.learning_rate)

        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs([loss, accuracy])

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value, accuracy_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, accuracy = %.2f, loss = %.2f (%1.f examples/sec; %.3f sec/batch)')
                    print(format_str % (
                    datetime.now(), self._step, accuracy_value, loss_value, examples_per_sec, sec_per_batch))

        conf = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement, allow_soft_placement=True,
                              intra_op_parallelism_threads=8)

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.ckpt,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                save_summaries_steps=FLAGS.save_steps,
                config=conf) as mon_sess:

            while not mon_sess.should_stop():
                mon_sess.run(train_op, {keep_prob: 0.7, batch_size: FLAGS.batch_size})


def evaluate_single():
    # PreProcess
    if not FLAGS.image:
        raise ValueError('Please supply a image')
    if tf.gfile.Exists(FLAGS.image) == False:
        raise ValueError('Image not found.')

    if tf.gfile.Exists(FLAGS.ckpt) == False:
        raise ValueError('Please supply a checkpoint')

    if FLAGS.num_of_classes == None:
        raise ValueError('Please supply num_of_classes.')

    if FLAGS.event_dir == None:
        raise ValueError('Please supply a event_dir')

    if tf.gfile.Exists(FLAGS.event_dir):
        tf.gfile.DeleteRecursively(FLAGS.event_dir)
    tf.gfile.MakeDirs(FLAGS.event_dir)

    with tf.Graph().as_default() as g:
        nn = CNN(FLAGS.network, FLAGS.num_of_classes, 1, FLAGS.image_size, FLAGS.image_crop_size, True, FLAGS.grayscale,
                 True, FLAGS.use_fp16)

        with tf.device("/cpu:0"):
            file_data = tf.read_file(FLAGS.image)
            image = tf.image.decode_jpeg(file_data, channels=3)
            image = nn.input(image, True, False)

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        keep_prob = tf.constant(1.0, dtype=dtype)
        batch_size = tf.constant(1)
        logits = nn.inference(image, keep_prob, batch_size)

        # Calculate predictions.
        logits = tf.cast(logits, tf.float32)
        softmax = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(CNN.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        info = [
            ['Image', str(FLAGS.image)],
            ['Image size', str(FLAGS.image_size)],
            ['Image Crop size', str(FLAGS.image_crop_size)],
            ['Grayscale', str(FLAGS.grayscale)],
            ['Use Float16', str(FLAGS.use_fp16)]
        ]
        info_summary = tf.summary.text('Info', tf.convert_to_tensor(info), collections=[])

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.event_dir, g)

        with tf.Session() as sess:
            summary_writer.add_summary(sess.run(info_summary))

            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            result = sess.run(softmax)
            result = result[0]

            print('- Result')
            for i, v in enumerate(result):
                print('[%d]: %f' % (i, v))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary_writer.add_summary(summary, global_step)


def evaluate():
    # PreProcess
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    if tf.gfile.Exists(FLAGS.ckpt) == False:
        raise ValueError('Please supply a checkpoint')

    if FLAGS.event_dir == None:
        raise ValueError('Please supply a event_dir')

    if tf.gfile.Exists(FLAGS.event_dir):
        tf.gfile.DeleteRecursively(FLAGS.event_dir)
    tf.gfile.MakeDirs(FLAGS.event_dir)

    num_of_classes = _num_of_folders(FLAGS.data_dir)
    if num_of_classes <= 0:
        raise ValueError('Invalid data_dir')

    num_of_samples = _num_of_files(FLAGS.data_dir)
    if num_of_samples == None:
        raise ValueError('Please supply num_of_samples.')

    print('[ SUMMARY ]')
    print('Num of classes: {}'.format(num_of_classes))
    print('Num of samples: {}'.format(num_of_samples))

    """Eval CIFAR-10 for a number of steps."""
    with tf.Graph().as_default() as g:
        info = [
            ['Title', FLAGS.title],
            ['Number of classes', str(num_of_classes)],
            ['Number of samples', str(num_of_samples)],
            ['Image size', str(FLAGS.image_size)],
            ['Image Crop size', str(FLAGS.image_crop_size)],
            ['Grayscale', str(FLAGS.grayscale)],
            ['Use Float16', str(FLAGS.use_fp16)]
        ]
        info_summary = tf.summary.text('NetworkInfo', tf.convert_to_tensor(info), collections=[])

        nn = CNN(FLAGS.network, num_of_classes, num_of_samples, FLAGS.image_size, FLAGS.image_crop_size,
                 FLAGS.log_input, FLAGS.grayscale, FLAGS.log_feature, FLAGS.use_fp16)

        with tf.device("/cpu:0"):
            # Get images and labels for CIFAR-10.
            images, labels, filenames = nn.inputs(FLAGS.data_dir, FLAGS.batch_size, FLAGS.batch_size * 100,
                                                  FLAGS.scaling, False)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        keep_prob = tf.constant(1.0, dtype=dtype)
        batch_size = tf.constant(FLAGS.batch_size)
        logits = nn.inference(images, keep_prob, batch_size)

        # Calculate predictions.
        logits = tf.cast(logits, tf.float32)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(CNN.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.event_dir, g)

        # eval_once(saver, summary_writer, top_k_op, summary_op, labels, keep_prob)
        with tf.Session() as sess:
            summary_writer.add_summary(sess.run(info_summary))

            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                # Assuming model_checkpoint_path looks something like:
                #    /my-favorite-path/cifar10_train/model.ckpt-0,
                # extract global_step from it.
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(num_of_samples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                # print(sess.run(labels, {keep_prob: 1.0}))

                errors = []
                while step < num_iter and not coord.should_stop():
                    predictions, labels_value, filenames_value = sess.run([top_k_op, labels, filenames])
                    true_count += np.sum(predictions)
                    step += 1

                    print(labels_value)
                    # print(predictions)
                    errors += [x for i, x in enumerate(filenames_value) if predictions[i] == False]

                # Print errors
                print('Errors:')
                print('\n'.join(f for f in errors))

                # Copy Errors
                copyErrors = True if FLAGS.error_dir != None else False
                if copyErrors:
                    _copy_errors(errors, FLAGS.error_dir)

                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, global_step)
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


def _copy_errors(errors, dest_root):
    if tf.gfile.Exists(dest_root):
        tf.gfile.DeleteRecursively(dest_root)

    for file_path in errors:
        label = os.path.basename(os.path.dirname(file_path))
        dest = os.path.join(dest_root, label)
        if tf.gfile.Exists(dest) == False:
            tf.gfile.MakeDirs(dest)

        dest = os.path.join(dest, os.path.basename(file_path))
        copyfile(file_path, dest)


def export():
    output_dir = os.path.join(FLAGS.ckpt, "export")
    if tf.gfile.Exists(output_dir) == True:
        tf.gfile.DeleteRecursively(output_dir)

    if tf.gfile.Exists(output_dir) == False:
        tf.gfile.MakeDirs(output_dir)

    ### EDIT TRAINING GRAPH ###
    with tf.Graph().as_default() as g:
        nn = CNN(FLAGS.network, FLAGS.num_of_classes, 1, FLAGS.image_size, FLAGS.image_crop_size, FLAGS.log_input,
                 FLAGS.grayscale, FLAGS.log_feature, FLAGS.use_fp16)

        with tf.device("/cpu:0"):
            image = tf.placeholder(tf.float32, shape=(None, None, 3), name="image")
            image = nn.input(image, True, False)

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        keep_prob = tf.constant(1.0, dtype=dtype)
        batch_size = tf.constant(1)
        logits = nn.inference(image, keep_prob, batch_size)

        # Calculate predictions.
        logits = tf.cast(logits, tf.float32)
        softmax = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(CNN.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        """
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(output_dir, g)
        """
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            tf.train.Saver().save(sess, os.path.join(output_dir, 'model.ckpt'),
                                  global_step=tf.convert_to_tensor(global_step))
            tf.train.write_graph(sess.graph.as_graph_def(), output_dir, 'graph.pbtxt', as_text=True)

            """
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary_writer.add_summary(summary, global_step)
            """

    ### EXPORT MODEL ###
    graph_path = os.path.join(output_dir, 'graph.pbtxt')
    if tf.gfile.Exists(graph_path) == False:
        raise ValueError('Graph not found({})'.format(graph_path))

    ckpt = tf.train.get_checkpoint_state(output_dir)
    ckpt_path = ckpt.model_checkpoint_path

    if ckpt == False or ckpt_path == False:
        raise ValueError('Check point not found.')

    output_path = os.path.join(output_dir, 'frozen.pb')
    optimized_output_path = os.path.join(output_dir, 'optimized.pb')

    freeze_graph.freeze_graph(input_graph=graph_path, input_saver="",
                              input_binary=False, input_checkpoint=ckpt_path,
                              output_node_names="softmax_linear/softmax",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0",
                              output_graph=output_path, clear_devices=True, initializer_nodes="")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open(output_path, "r") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ['image'],
        ["softmax_linear/softmax"],
        tf.float32.as_datatype_enum)

    f = tf.gfile.FastGFile(optimized_output_path, "w")
    f.write(output_graph_def.SerializeToString())

    output_size = os.path.getsize(output_path)
    optimized_output_size = os.path.getsize(optimized_output_path)

    print('Model Exported successfuly.')
    print('- Frozen Model: {} ({})'.format(output_path, _humansize(output_size)))
    print('- Optimized Model: {} ({})'.format(optimized_output_path, _humansize(optimized_output_size)))


def evaluate_with_pb():
    # PreProcess
    if not FLAGS.image:
        raise ValueError('Please supply a image')
    if tf.gfile.Exists(FLAGS.image) == False:
        raise ValueError('Image not found.')

    if tf.gfile.Exists(FLAGS.pb) == False:
        raise ValueError('Please supply a pb')

    im = Image.open(FLAGS.image)
    image_data = np.array(im.getdata()).reshape([im.width, im.height, 3])

    image_data = [v / 255.0 for v in image_data]
    with tf.Graph().as_default() as g:
        with tf.Session() as new_sess:
            with tf.gfile.FastGFile(FLAGS.pb, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

            softmax = new_sess.graph.get_tensor_by_name("softmax_linear/softmax:0")

            # Loading the injected placeholder
            input_placeholder = new_sess.graph.get_tensor_by_name("image:0")

            result = new_sess.run(softmax, {input_placeholder: image_data})[0]
            print('- Result')
            for i, v in enumerate(result):
                print('[%d]: %f' % (i, v))


def export_serving():
    output_dir = os.path.join(FLAGS.ckpt, "export_serving")
    if tf.gfile.Exists(output_dir) == True:
        tf.gfile.DeleteRecursively(output_dir)

    if tf.gfile.Exists(output_dir) == False:
        tf.gfile.MakeDirs(output_dir)

    ### EDIT TRAINING GRAPH ###
    with tf.Graph().as_default() as g:
        nn = CNN(FLAGS.network, FLAGS.num_of_classes, 1, FLAGS.image_size, FLAGS.image_crop_size, FLAGS.log_input,
                 FLAGS.grayscale, FLAGS.log_feature, FLAGS.use_fp16)

        with tf.device("/cpu:0"):
            image = tf.placeholder(tf.float32, shape=(None, None, 3), name="image")
            image = nn.input(image, True, False)

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        keep_prob = tf.constant(1.0, dtype=dtype)
        batch_size = tf.constant(1)
        logits = nn.inference(image, keep_prob, batch_size)

        # Calculate predictions.
        logits = tf.cast(logits, tf.float32)
        softmax = tf.nn.softmax(logits)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(CNN.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        """
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(output_dir, g)
        """
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            else:
                print('No checkpoint file found')
                return

            tf.train.Saver().save(sess, os.path.join(output_dir, 'model.ckpt'),
                                  global_step=tf.convert_to_tensor(global_step))
            tf.train.write_graph(sess.graph.as_graph_def(), output_dir, 'graph.pbtxt', as_text=True)

            """
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary_writer.add_summary(summary, global_step)
            """

            ### EXPORT MODEL ###
            export_path_base = output_dir
            export_path = os.path.join(tf.compat.as_bytes(export_path_base),
                                       tf.compat.as_bytes(str(FLAGS.model_version)))
            print('Exporting model to '.format(export_path))

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # Build the signature_def_map.
            classification_inputs = tf.saved_model.utils.build_tensor_info(image)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(softmax)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(softmax)

            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS: classification_inputs
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

            tensor_info_x = tf.saved_model.utils.build_tensor_info(image)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(softmax)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'image': tensor_info_x},
                    outputs={'score': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_image': prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()

            print('Done exporting!')


def _humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])


class Classifier(object):
    def __init__(self, pb, out_node_name):
        with tf.Graph().as_default() as g:
            with tf.Session() as new_sess:
                self.new_sess = new_sess
                with tf.gfile.FastGFile(pb, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(graph_def, name='')

                # self.softmax = new_sess.graph.get_tensor_by_name("softmax_linear/softmax:0")
                self.out_weights = new_sess.graph.get_tensor_by_name(out_node_name)

                # Loading the injected placeholder
                self.image = new_sess.graph.get_tensor_by_name("image:0")

    def regression(self, filepath):
        image_data = self._get_image_data(filepath)
        image_data = [v / 255.0 for v in image_data]

        result = self.new_sess.run(self.out_weights, {self.image: image_data})[0]
        return result.flatten().tolist()

    def _get_image_data(self, file_path):
        with Image.open(file_path) as im:
            with self._to_squared_image(im) as squared_img:
                return np.array(squared_img.getdata()).reshape([squared_img.width, squared_img.height, 3])

    def _to_squared_image(self, img):
        longer_side = max(img.size)
        horizontal_padding = (longer_side - img.size[0]) / 2
        vertical_padding = (longer_side - img.size[1]) / 2
        return img.crop(
            (
                -horizontal_padding,
                -vertical_padding,
                img.size[0] + horizontal_padding,
                img.size[1] + vertical_padding
            )
        )


def map_export(pb_file_path, feature_node_name, data_dir, map_file):
    classifier = Classifier(pb_file_path, feature_node_name)

    with open(map_file, 'wb') as fp:
        for (path, dir, files) in os.walk(data_dir):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext in IMAGE_FILE_EXTS:
                    file_path = os.path.join(data_dir, path, filename)
                    label = os.path.splitext(filename)[0]

                    features = classifier.regression(file_path)
                    pickle.dump([label, features], fp)

                    print(filename)


def map_inference(pb_file_path, feature_node_name, image_file_path, map_file):
    classifier = Classifier(pb_file_path, feature_node_name)

    with open(map_file, 'rb') as fp:
        features = classifier.regression(image_file_path)

        min_value = 100000
        min_label = 'None'
        try:
            while True:
                row = pickle.load(fp)

                row_features = np.array(row[1])
                features = np.array(features)
                dist = np.linalg.norm(row_features - features)
                if min_value > dist:
                    min_value = dist
                    min_label = row[0]
        except EOFError:
            pass

        print('Winner: {}, {}'.format(min_value, min_label))


def main(argv=None):
    if FLAGS.mode == 'train':
        if FLAGS.network == None:
            raise ValueError('Please supply a network(graph) file.')

        if FLAGS.ckpt == None:
            raise ValueError('Please supply a ckpt dir.')

        if FLAGS.data_dir == None:
            raise ValueError('Please supply a data_dir.')

        if FLAGS.continue_train != True and tf.gfile.Exists(FLAGS.ckpt):
            tf.gfile.DeleteRecursively(FLAGS.ckpt)

        if tf.gfile.Exists(FLAGS.ckpt) == False:
            tf.gfile.MakeDirs(FLAGS.ckpt)

        # WRITE META-DATA
        FLAGS.num_of_classes = _num_of_folders(FLAGS.data_dir)
        with open(os.path.join(FLAGS.ckpt, "flags.txt"), "w") as text_file:
            flags = json.dumps(FLAGS.__dict__["__flags"])
            text_file.write(flags)

        train()

    elif FLAGS.mode == 'eval':
        if FLAGS.ckpt == None:
            raise ValueError('Please supply a ckpt dir.')

        if FLAGS.data_dir == None:
            raise ValueError('Please supply a data_dir.')

        if FLAGS.ckpt == FLAGS.event_dir:
            raise ValueError('event_dir cannot be same with ckpt dir!!!')

        # READ META-DATA
        with open(os.path.join(FLAGS.ckpt, "flags.txt"), "r") as text_file:
            txt = text_file.read()
            flags = json.loads(txt)
            FLAGS.num_of_classes = flags['num_of_classes']
            FLAGS.image_size = flags['image_size']
            FLAGS.image_crop_size = flags['image_crop_size']
            FLAGS.grayscale = flags['grayscale']
            FLAGS.use_fp16 = flags['use_fp16']
            FLAGS.network = flags['network']
            FLAGS.learning_rate = flags['learning_rate']

        evaluate()

    elif FLAGS.mode == 'export':
        if FLAGS.ckpt == None:
            raise ValueError('Please supply a ckpt dir.')

        # READ META-DATA
        with open(os.path.join(FLAGS.ckpt, "flags.txt"), "r") as text_file:
            txt = text_file.read()
            flags = json.loads(txt)
            FLAGS.num_of_classes = flags['num_of_classes']
            FLAGS.image_size = flags['image_size']
            FLAGS.image_crop_size = flags['image_crop_size']
            FLAGS.grayscale = flags['grayscale']
            FLAGS.use_fp16 = flags['use_fp16']
            FLAGS.network = flags['network']
            FLAGS.learning_rate = flags['learning_rate']

        export()

    elif FLAGS.mode == 'export-serving':
        if FLAGS.ckpt == None:
            raise ValueError('Please supply a ckpt dir.')

        # READ META-DATA
        with open(os.path.join(FLAGS.ckpt, "flags.txt"), "r") as text_file:
            txt = text_file.read()
            flags = json.loads(txt)
            FLAGS.num_of_classes = flags['num_of_classes']
            FLAGS.image_size = flags['image_size']
            FLAGS.image_crop_size = flags['image_crop_size']
            FLAGS.grayscale = flags['grayscale']
            FLAGS.use_fp16 = flags['use_fp16']
            FLAGS.network = flags['network']
            FLAGS.learning_rate = flags['learning_rate']

        export_serving()


    elif FLAGS.mode == 'single':
        if FLAGS.ckpt == None:
            raise ValueError('Please supply a ckpt dir.')

        if FLAGS.image == None:
            raise ValueError('Please supply a image file.')

        # READ META-DATA
        with open(os.path.join(FLAGS.ckpt, "flags.txt"), "r") as text_file:
            txt = text_file.read()
            flags = json.loads(txt)
            FLAGS.num_of_classes = flags['num_of_classes']
            FLAGS.image_size = flags['image_size']
            FLAGS.image_crop_size = flags['image_crop_size']
            FLAGS.grayscale = flags['grayscale']
            FLAGS.use_fp16 = flags['use_fp16']
            FLAGS.network = flags['network']
            FLAGS.learning_rate = flags['learning_rate']

        evaluate_single()

    elif FLAGS.mode == 'pb':
        if FLAGS.pb == None:
            raise ValueError('Please supply a pb file.')

        if FLAGS.image == None:
            raise ValueError('Please supply a image file.')

        evaluate_with_pb();


    elif FLAGS.mode == 'resize':
        if FLAGS.data_dir == None:
            raise ValueError('Please supply a data_dir.')

        if FLAGS.out_dir == None:
            raise ValueError('Please supply a out_dir.')

        if tf.gfile.Exists(FLAGS.out_dir):
            tf.gfile.DeleteRecursively(FLAGS.out_dir)

        tf.gfile.MakeDirs(FLAGS.out_dir)

        convert_count = CNN.convert_img_to_square_with_pad(FLAGS.data_dir, FLAGS.out_dir)
        print('Complete Succefuly.')
        print('Output Dir: {}'.format(FLAGS.out_dir))
        print('Total Count: {}'.format(convert_count))


    elif FLAGS.mode == 'map-export':
        if FLAGS.pb == None:
            raise ValueError('Please supply pb.')
        if tf.gfile.Exists(FLAGS.pb) == False:
            raise ValueError('Pb not exists.')

        if FLAGS.out_node == None:
            raise ValueError('Please supply out_node.')

        if FLAGS.data_dir == None:
            raise ValueError('Please supply data_dir.')
        if tf.gfile.Exists(FLAGS.data_dir) == False:
            raise ValueError('data_dir not exists.')

        if FLAGS.map == None:
            raise ValueError('Please supply map.')
        if tf.gfile.Exists(FLAGS.map) == True:
            tf.gfile.Remove(FLAGS.map)

        map_export(FLAGS.pb, FLAGS.out_node, FLAGS.data_dir, FLAGS.map)


    elif FLAGS.mode == 'map-inference':
        if FLAGS.pb == None:
            raise ValueError('Please supply pb.')
        if tf.gfile.Exists(FLAGS.pb) == False:
            raise ValueError('Pb not exists.')

        if FLAGS.out_node == None:
            raise ValueError('Please supply out_node.')

        if FLAGS.image == None:
            raise ValueError('Please supply image.')
        if tf.gfile.Exists(FLAGS.image) == False:
            raise ValueError('image not exists.')

        if FLAGS.map == None:
            raise ValueError('Please supply map.')
        if tf.gfile.Exists(FLAGS.map) == False:
            raise ValueError('Map not exists.')

        map_inference(FLAGS.pb, FLAGS.out_node, FLAGS.image, FLAGS.map)


if __name__ == '__main__':
    tf.app.run()
