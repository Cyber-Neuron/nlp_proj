# coding=utf-8
'''
Created on Dec 6, 2018

@author: dan
'''

#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import features_to_nonpadding, fast_decode_tpu,fast_decode,transformer_prepare_decoder
import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS

@registry.register_model
class TransferTransformer(transformer.Transformer):
    """Transfer learning Transformer"""
    def __init__(self, *args, **kwargs):
        super(TransferTransformer, self).__init__(*args, **kwargs)

#     @staticmethod
#     def eval_hooks():
#         new_model_scope="transfer_transformer"
#         old_model_scope="transfer_transformer"
#         exclude=["class_label_modality_","symbol_modality_"]
#         restore_resnet_hook = RestoreHook(
#           # TODO(zichaoy): hard code the path given static function.
#           checkpoint_path="/home/liudan/pyz-403-aa/PAN/t2t/outbak",#"/home/zichaoy/resnet_v1_152.ckpt",
#           exclude=["class_label_modality_"]#,"symbol_modality_"]
#       )
#         if tf.train.latest_checkpoint(FLAGS.output_dir) is None:
#             return []
# #         variables_to_restore = tf.contrib.framework.get_variables_to_restore(
# #         include=None, exclude=exclude)
# #         print("000;",variables_to_restore)
# #         # remove new_model_scope from variable name prefix
# #         assignment_map = {variable.name[len(new_model_scope):]: variable
# #                           for variable in variables_to_restore
# #                           if variable.name.startswith(new_model_scope)}
# #         # remove :0 from variable name suffix
# #         assignment_map = {name.split(":")[0]: variable
# #                           for name, variable in six.iteritems(assignment_map)
# #                           if name.startswith(old_model_scope)}
# #         print("111;",assignment_map)
# #         _=1/0
#         return [restore_resnet_hook]
#     @staticmethod
#     def train_hooks():
#         new_model_scope="transfer_transformer"
#         old_model_scope="transfer_transformer"
#         exclude=["class_label_modality_","symbol_modality_"]
#         restore_resnet_hook = RestoreHook(
#           # TODO(zichaoy): hard code the path given static function.
#           checkpoint_path="/home/liudan/pyz-403-aa/PAN/t2t/outbak",#"/home/zichaoy/resnet_v1_152.ckpt",
#           exclude=["class_label_modality_"]#,"symbol_modality_"]
#       )
#         if tf.train.latest_checkpoint(FLAGS.output_dir) is None:
#             return []
# #         variables_to_restore = tf.contrib.framework.get_variables_to_restore(
# #         include=None, exclude=exclude)
# #         print("000;",variables_to_restore)
# #         # remove new_model_scope from variable name prefix
# #         assignment_map = {variable.name[len(new_model_scope):]: variable
# #                           for variable in variables_to_restore
# #                           if variable.name.startswith(new_model_scope)}
# #         # remove :0 from variable name suffix
# #         assignment_map = {name.split(":")[0]: variable
# #                           for name, variable in six.iteritems(assignment_map)
# #                           if name.startswith(old_model_scope)}
# #         print("111;",assignment_map)
# #         _=1/0
#         return [restore_resnet_hook]
    def initialize_from_ckpt(self, ckpt_dir):
        model_dir = self._hparams.get("model_dir", None)
        already_has_ckpt = (
            model_dir and tf.train.latest_checkpoint(model_dir) is not None)
        if already_has_ckpt:
            return
        
        restore = RestoreHook(
          # TODO(zichaoy): hard code the path given static function.
          checkpoint_path=ckpt_dir,#"/home/zichaoy/resnet_v1_152.ckpt",
          exclude=["class_label_modality_"]#,"symbol_modality_"]
      )
        restore.begin()
    def encode(self, inputs, target_space, hparams, features=None, losses=None):
        return super().encode(inputs, target_space, hparams, features, losses)

    def decode(self,
               decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               hparams,
               cache=None,
               decode_loop_step=None,
               nonpadding=None,
               losses=None):

        return super().decode(decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               hparams,
               cache,
               decode_loop_step,
               nonpadding,
               losses)

    def body(self, features):
        """Transformer main model_fn.

        Args:
          features: Map of features to the model. Should contain the following:
              "inputs": Transformer inputs.
                  [batch_size, input_length, 1, hidden_dim].
              "targets": Target decoder outputs.
                  [batch_size, decoder_length, 1, hidden_dim]
              "target_space_id": A scalar int from data_generators.problem.SpaceID.

        Returns:
          Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        #self._hparams.add("warm_start_from",True)
        hparams = self._hparams

        losses = []

#         if self.has_input:
#             inputs = features["inputs"]
#             target_space = features["target_space_id"]
#             encoder_output, encoder_decoder_attention_bias = self.encode(
#                 inputs, target_space, hparams, features=features, losses=losses)
#         else:
        encoder_output, encoder_decoder_attention_bias = (None, None)
        lekeys="inputs"
        if lekeys in features:
            targets = features["inputs"]
            lekeys="inputs"
        else:
            targets = features["targets"]
            lekeys="targets"
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
            targets, hparams, features=features)
        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=features_to_nonpadding(features, lekeys),
            losses=losses)

        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        ret = tf.reshape(decoder_output, targets_shape)
        if losses:
            return ret, {"extra_loss": tf.add_n(losses)}
        else:
            return ret

    def _greedy_infer(self, features, decode_length, use_tpu=False):
        """Fast version of greedy decoding.

        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          use_tpu: A bool. Whether to build the inference graph for TPU.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        # For real-valued modalities use the slow decode path for now.
        if (self._target_modality_is_real or
            self._hparams.self_attention_type != "dot_product"):
            return  super(TransferTransformer, self)._greedy_infer(features, decode_length)
        with tf.variable_scope(self.name):
            return (self._fast_decode_tpu(features, decode_length) if use_tpu else
                    self._fast_decode(features, decode_length))

    def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha):
        """Beam search decoding.

        Args:
          features: an map of string to `Tensor`
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }
        """
        if self._hparams.self_attention_type != "dot_product":
            # Caching is not guaranteed to work with attention types other than
            # dot_product.
            # TODO(petershaw): Support fast decoding when using relative
            # position representations, i.e. "dot_product_relative" attention.
            return self._beam_decode_slow(features, decode_length, beam_size,
                                          top_beams, alpha)
        with tf.variable_scope(self.name):
            return self._fast_decode(features, decode_length, beam_size, top_beams,
                                     alpha)

    def _fast_decode_tpu(self,
                         features,
                         decode_length,
                         beam_size=1):
        """Fast decoding.

        Implements only greedy decoding on TPU.

        Args:
          features: A map of string to model features.
          decode_length: An integer, how many additional timesteps to decode.
          beam_size: An integer, number of beams.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }.

        Raises:
          NotImplementedError: If there are multiple data shards or beam_size > 1.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.target_modality

        if self.has_input:
            inputs = features["inputs"]
            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = (
                    common_layers.shape_list(inputs)[1] + features.get(
                        "decode_length", decode_length))

            # TODO(llion): Clean up this reshaping logic.
            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.input_modality["inputs"]
            with tf.variable_scope(input_modality.name):
                inputs = input_modality.bottom_sharded(inputs, dp)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length + 1, hparams.hidden_size]),
                hparams.max_length, "body/targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: A tensor, inputs ids to the decoder. [batch_size, 1].
              i: An integer, Step number of the decoding loop.

            Returns:
              A tensor, processed targets [batch_size, 1, hidden_dim].
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            # TODO(llion): Explain! Is this even needed?
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                positional_encoding_shape = positional_encoding.shape.as_list()
                targets += tf.slice(
                    positional_encoding, [0, i, 0],
                    [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_tpu_fn(ids, i, cache):
            """Go from ids to logits for next symbol on TPU.

            Args:
              ids: A tensor, symbol IDs.
              i: An integer, step number of the decoding loop. Only used for inference
                  on TPU.
              cache: A dict, containing tensors which are the results of previous
                  attentions, used for fast decoding.

            Returns:
              ret: A tensor, computed logits.
              cache: A dict, containing tensors which are the results of previous
                  attentions, used for fast decoding.
            """
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias_shape = decoder_self_attention_bias.shape.as_list()
            bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                            [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    i,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(
                            tf.slice(partial_targets, [0, i],
                                     [partial_targets.shape.as_list()[0], 1]),
                            [beam_size]), vocab_size, 0.0, -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = fast_decode_tpu(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_tpu_fn,
            hparams=hparams,
            decode_length=decode_length,
            beam_size=beam_size,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            ret["outputs"] = ret["outputs"][:, partial_targets_length:]
        return ret

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        """Fast decoding.

        Implements both greedy and beam search decoding, uses beam search iff
        beam_size > 1, otherwise beam search related arguments are ignored.

        Args:
          features: a map of string to model  features.
          decode_length: an integer.  How many additional timesteps to decode.
          beam_size: number of beams.
          top_beams: an integer. How many of the beams to return.
          alpha: Float that controls the length penalty. larger the alpha, stronger
            the preference for longer translations.

        Returns:
          A dict of decoding results {
              "outputs": integer `Tensor` of decoded ids of shape
                  [batch_size, <= decode_length] if beam_size == 1 or
                  [batch_size, top_beams, <= decode_length]
              "scores": decoding log probs from the beam search,
                  None if using greedy decoding (beam_size=1)
          }

        Raises:
          NotImplementedError: If there are multiple data shards.
        """
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.target_modality
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        if self.has_input:
            inputs = features["inputs"]
            if target_modality.is_class_modality:
                decode_length = 1
            else:
                decode_length = (
                    common_layers.shape_list(inputs)[1] + features.get(
                        "decode_length", decode_length))

            # TODO(llion): Clean up this reshaping logic.
            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.input_modality["inputs"]
            with tf.variable_scope(input_modality.name):
                inputs = input_modality.bottom_sharded(inputs, dp)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length, hparams.hidden_size]),
                hparams.max_length, "body/targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.

            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            with tf.variable_scope(target_modality.name):
                targets = target_modality.targets_bottom_sharded(targets, dp)[0]
            targets = common_layers.flatten4d3d(targets)

            # TODO(llion): Explain! Is this even needed?
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                targets += positional_encoding[:, i:i + 1]
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    nonpadding=features_to_nonpadding(features, "targets"))

            with tf.variable_scope(target_modality.name):
                logits = target_modality.top_sharded(body_outputs, None, dp)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
                        -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_modality.top_dimensionality,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
        return ret



@registry.register_hparams
def transformer_transf():
    hparams = transformer.transformer_base()
    hparams.hidden_size = 128
    hparams.filter_size = 128
    hparams.num_heads = 2
    return hparams
import six
class RestoreHook(tf.train.SessionRunHook):
    """Restore variables from a checkpoint path."""

    def __init__(self, checkpoint_path="", new_model_scope="", old_model_scope="",
               include=None, exclude=None):
        self._checkpoint_path = checkpoint_path
        self._new_model_scope = new_model_scope
        self._old_model_scope = old_model_scope
        self._include = include
        self._exclude = exclude
        self._assignment_map={}
        self._exclued=[]

    def begin(self):
        """Load variables from checkpoint.

        New model variables have the following name foramt:
        new_model_scope/old_model_scope/xxx/xxx:0 To find the map of
        name to variable, need to strip the new_model_scope and then
        match the old_model_scope and remove the suffix :0.

        """
        variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        # remove new_model_scope from variable name prefix
#         assignment_map = {variable.name[len(self._new_model_scope):]: variable
#                           for variable in variables_to_restore
#                           if variable.name.startswith(self._new_model_scope)}
#         # remove :0 from variable name suffix
#         assignment_map = {name.split(":")[0]: variable
#                           for name, variable in six.iteritems(assignment_map)
#                           if name.startswith(self._old_model_scope)}
        excludes=self._exclude
        for variable in variables_to_restore:
            skip=False
            for exc in excludes:
                if exc in variable.name:
                    skip=True
                    break
            if skip:
                self._exclued.append(variable.name)
                continue
            self._assignment_map[variable.name]=variable
        tf.logging.info("restoring %d variables from checkpoint %s"%(
            len(self._assignment_map), self._checkpoint_path))
        self._assignment_map = {name.split(":")[0]: variable
                          for name, variable in six.iteritems(self._assignment_map)
                          if name.startswith(self._old_model_scope)}
        tf.train.init_from_checkpoint(self._checkpoint_path, self._assignment_map)
        print("Done")
    def getrv(self,variables_to_restore):
        _assignment_map=[]
        excludes=self._exclude
        for variable in variables_to_restore:
            skip=False
            for exc in excludes:
                if exc in variable.name:
                    skip=True
                    break
            if skip:
                continue
            _assignment_map.append(variable)
        return _assignment_map
    def begin2(self):
        """Load variables from checkpoint.
    
        New model variables have the following name foramt:
        new_model_scope/old_model_scope/xxx/xxx:0 To find the map of
        name to variable, need to strip the new_model_scope and then
        match the old_model_scope and remove the suffix :0.
    
        """
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            include=self._include, exclude=self._exclude)
        variables_to_restore = self.getrv(variables_to_restore)
        # remove new_model_scope from variable name prefix
        assignment_map = {variable.name[len(self._new_model_scope):]: variable
                          for variable in variables_to_restore
                          if variable.name.startswith(self._new_model_scope)}
        print("000:",assignment_map)
        # remove :0 from variable name suffix
        assignment_map = {name.split(":")[0]: variable
                          for name, variable in six.iteritems(assignment_map)
                          if name.startswith(self._old_model_scope)}
        self._assignment_map = assignment_map
        tf.logging.info("restoring %d variables from checkpoint %s"%(
            len(assignment_map), self._checkpoint_path))
        tf.train.init_from_checkpoint(self._checkpoint_path, self._assignment_map)
        