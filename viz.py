import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import collections

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# Setup some directories
data_dir = os.path.expanduser("data")
tmp_dir = os.path.expanduser("tmp")
train_dir = os.path.expanduser("train")
checkpoint_dir = os.path.expanduser("output")
# tf.gfile.MakeDirs(data_dir)
# tf.gfile.MakeDirs(tmp_dir)
#tf.gfile.MakeDirs(train_dir)
# tf.gfile.MakeDirs(checkpoint_dir)
gs_data_dir = "data"
gs_ckpt_dir = "checkpoints"

ende_problem = problems.problem("sentiment_imdb")
# Fetch the problem

# Copy the vocab file locally so we can encode inputs and decode model outputs
# All vocabs are stored on GCS
vocab_name = "vocab.sentiment_imdb.8192.subwords"
vocab_file = os.path.join(gs_data_dir, vocab_name)

# Get the encoders from the problem
encoders = ende_problem.feature_encoders(data_dir)
print encoders
# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}



model_name = "transformer"
hparams_set = "transformer_tiny"

hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name="sentiment_imdb")

# NOTE: Only create the model once when restoring from a checkpoint; it's a
# Layer and so subsequent instantiations will have different variable scopes
# that will not match the checkpoint.
translate_model = registry.model(model_name)(hparams, Modes.EVAL)
ckpt_name = ""
gs_ckpt = os.path.join(gs_ckpt_dir, ckpt_name)
#!gsutil -q cp -R {gs_ckpt} {checkpoint_dir}
ckpt_path = tf.train.latest_checkpoint(os.path.join(checkpoint_dir, ckpt_name))
print ckpt_path
def translate(inputs):
    encoded_inputs = encode(inputs)
    with tfe.restore_variables_on_create(ckpt_path):
        model_output = translate_model.infer(encoded_inputs)["outputs"]
    return translate_model(encoded_inputs)

inputs = "A beauty day with sad mood 0BAD 0GOOD"
#outputs = translate(inputs)

print("Inputs: %s" % inputs)
# print("Outputs: %s" % outputs)
from tensor2tensor.visualization import attention
from tensor2tensor.data_generators import text_encoder

SIZE = 35

def encode_eval(input_str):
    inputs = tf.reshape(encoders["inputs"].encode(input_str) + [1], [1, -1, 1, 1])  # Make it 3D.
#     outputs = tf.reshape(encoders["inputs"].encode(output_str) + [1], [1, -1, 1, 1])  # Make it 3D.
    return {"inputs": inputs, "targets": inputs}

def get_att_mats():
    enc_atts = []
    #dec_atts = []
    encdec_atts = []

    for i in range(hparams.num_hidden_layers):
        enc_att = translate_model.attention_weights[
          "transformer/body/encoder/layer_%i/self_attention/multihead_attention/dot_product_attention" % i][0]
        #dec_att = translate_model.attention_weights[
        #"transformer/body/decoder/layer_%i/self_attention/multihead_attention/dot_product_attention" % i][0]
#         encdec_att = translate_model.attention_weights[
#           "transformer/body/decoder/layer_%i/encdec_attention/multihead_attention/dot_product_attention" % i][0]
        enc_atts.append(resize(enc_att))
        #dec_atts.append(resize(dec_att))
#         encdec_atts.append(resize(encdec_att))
    return enc_atts

def resize(np_mat):
    # Sum across heads
    np_mat = np_mat[:, :SIZE, :SIZE]
    row_sums = np.sum(np_mat, axis=0)
    # Normalize
    layer_mat = np_mat / row_sums[np.newaxis, :]
    lsh = layer_mat.shape
    # Add extra dim for viz code to work.
    layer_mat = np.reshape(layer_mat, (1, lsh[0], lsh[1], lsh[2]))
    return layer_mat

def to_tokens(ids):
    ids = np.squeeze(ids)
    subtokenizer = hparams.problem_hparams.vocabulary['inputs']
    tokens = []
    for _id in ids:
        if _id == 0:
            tokens.append('<PAD>')
        elif _id == 1:
            tokens.append('<EOS>')
        elif _id == -1:
            tokens.append('<NULL>')
        else:
            tokens.append(subtokenizer._subtoken_id_to_subtoken_string(_id))
    return tokens
def call_html():
    import IPython
#     display(IPython.core.display.HTML('''
#         <script src="/static/components/requirejs/require.js"></script>
#         <script>
#           requirejs.config({
#             paths: {
#               base: '/static/base',
#               "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
#               jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
#             },
#           });
#         </script>
#         '''))
# Convert inputs and outputs to subwords
inp_text = to_tokens(encoders["inputs"].encode(inputs))
#out_text = to_tokens(encoders["inputs"].encode(outputs))

# Run eval to collect attention weights
example = encode_eval(inputs)
with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_dir)):
    translate_model.set_mode(Modes.EVAL)
    translate_model(example)
# Get normalized attention weights for each layer
enc_atts, encdec_atts = get_att_mats()
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.ticker as ticker
fig = plt.figure()



for i,z in enumerate(enc_atts[0]):
    fig, ax = plt.subplots()
    #ax = fig.add_subplot(2,2,i+1)
    nx, ny = z.shape
    indx, indy = np.arange(nx), np.arange(ny)
    x, y = np.meshgrid(indx, indy)
    
    
    ax.imshow(z.T, interpolation="nearest", cmap=cm.YlGn) # plot grid values
    
    for xval, yval in zip(x.flatten(), y.flatten()):
        zval = z[xval, yval]
        t = "%.1f%%"%(zval * 100,) # format value with 1 decimal point
        c = 'w' if zval > 0.75 else 'k' # if dark-green, change text color to white
        ax.text(xval, yval, t, color=c, va='center', ha='center')
    
    xlabels = inp_text +[""]
    ylabels = xlabels#xlabels[::-1] #xlabels
    ax.set_xticks(indx+0.5) # offset x/y ticks so gridlines run on border of boxes
    ax.set_yticks(indy+0.5)
    ax.grid(ls='-', lw=2)
    
    # the tick labels, if you want them centered need to be adjusted in 
    # this special way.
    for a, ind, labels in zip((ax.xaxis, ax.yaxis), (indx, indy), 
                              (xlabels, ylabels)):
        a.set_major_formatter(ticker.NullFormatter())
        a.set_minor_locator(ticker.FixedLocator(ind))
        a.set_minor_formatter(ticker.FixedFormatter(labels))
    
    ax.xaxis.tick_top()

    plt.show()
#call_html()
#attention.show(inp_text, inp_text, enc_atts, enc_atts, encdec_atts)