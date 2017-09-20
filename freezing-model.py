import os
import tensorflow as tf
from tensorflow.python.framework import graph_util

'''
ref: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
'''

def print_graph_names(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()


    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        names = [n.name for n in input_graph_def.node]

        for n in names:
            print(n)

    return


def freeze_graph(model_folder, output_pb_fn, export_names=None):
    if export_names is None:
        raise ValueError('Please state what nodes should be exported in export_names!!')

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    output_graph = os.path.join(model_folder, '{:s}.pb'.format(output_pb_fn))

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We import the meta graph and retrieve a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            input_graph_def,  # The graph_def is used to retrieve the nodes
            export_names  # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

        print('{:d} ops in the final graph.'.format(len(output_graph_def.node)))

    return


def load_graph(frozen_graph_filename, prefix='prefix'):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=prefix,
            op_dict=None,
            producer_op_list=None
        )
    return graph


def main():
    program_options = 'view_graph_names' # 'view_graph_names', 'freeze_graph', 'load_graph'

    ckpt_loc = '/path/to/ckpt/file/'
    if program_options == 'view_graph_names':
        print_graph_names(ckpt_loc)
    elif program_options == 'freeze_graph':
        output_pb_fn = 'freezed_model'
        export_list = []
        freeze_graph(ckpt_loc, output_pb_fn, export_list)
    else: # 'load_graph'
        pb_file = ''
        prefix = 'moo'
        graph = load_graph(pb_file, prefix)

    return


if __name__ == '__main__':
    main()