import tensorflow as tf

"""
From: https://stackoverflow.com/questions/45382917/how-to-optimize-for-inference-a-simple-saved-tensorflow-1-0-1-graph
"""

def saving():
    # make and save a simple graph
    tf.reset_default_graph()
    # G = tf.Graph()
    # with G.as_default():
    x = tf.placeholder(dtype=tf.float32, shape=(), name="x")
    a = tf.Variable(5.0, name="a")
    y = tf.add(a, x, name="y")
    saver = tf.train.Saver()

    # with tf.Session(graph=G) as sess:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out = sess.run(y, feed_dict={x: 1.0})

        # Save GraphDef
        tf.train.write_graph(sess.graph_def, './saved_model', 'graph.pb')

        #  Save checkpoint
        saver.save(sess=sess, save_path="./saved_model/my_model.ckpt")
    return


def freezing(input_graph, input_checkpoint, output_graph, output_node_names):
    from subprocess import call
    call(['python', '-m', 'tensorflow.python.tools.freeze_graph',
          '--input_graph', input_graph,
          '--input_checkpoint', input_checkpoint,
          '--output_graph', output_graph,
          '--output_node_names={:s}'.format(output_node_names)])
    return


def optimizing(input_frozen, output_optimized, input_names, output_names):
    # python -m tensorflow.python.tools.optimize_for_inference --input graph_frozen.pb --output graph_optimized.pb --input_names=x --output_names=y
    from subprocess import call
    call(['python', '-m', 'tensorflow.python.tools.optimize_for_inference',
          '--input', input_frozen,
          '--output', output_optimized,
          '--input_names={:s}'.format(input_names),
          '--output_names={:s}'.format(output_names)])
    return


def using_optimized(freezed_optimized):
    with tf.gfile.GFile(freezed_optimized, 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    G = tf.Graph()

    with tf.Session(graph=G) as sess:
        y, = tf.import_graph_def(graph_def_optimized, return_elements=['y:0'])
        print('Operations in Optimized Graph:')
        print([op.name for op in G.get_operations()])
        x = G.get_tensor_by_name('import/x:0')
        tf.global_variables_initializer().run()
        out = sess.run(y, feed_dict={x: 1.0})
        print(out)

    # Output
    # Operations in Optimized Graph:
    # ['import/x', 'import/a', 'import/y']
    # 6.0
    return


def main():
    # # step 1: save the model
    # saving()

    # # step 2: freeze the model
    # input_graph = './saved_model/graph.pb'
    # input_checkpoint = './saved_model/my_model.ckpt'
    # output_graph = './saved_model/graph_frozen.pb'
    # output_node_names = 'y'  # The name of the output nodes, comma separated
    # freezing(input_graph, input_checkpoint, output_graph, output_node_names)

    # # step 3: optimize the model
    # input_frozen = './saved_model/graph_frozen.pb'
    # output_optimized = './saved_model/graph_frozen_optimized.pb'
    # input_names = 'x'
    # output_names = 'y'
    # optimizing(input_frozen, output_optimized, input_names, output_names)

    # step 4: load the optimized model and do inference
    freezed_optimized = './saved_model/graph_frozen_optimized.pb'
    using_optimized(freezed_optimized)
    return


if __name__ == '__main__':
    main()
