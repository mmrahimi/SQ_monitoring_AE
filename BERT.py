import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
import tensorflow as tf
import tensorflow_hub as hub
import os
import collections
from InputExample import *
from InputFeatures import *
from PaddingInputExample import *
from datetime import datetime
import pandas as pd
import numpy as np
import skmultilearn
from skmultilearn.problem_transform import LabelPowerset
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

class BERT:
    def __init__(self, bert_path, output_path):
        self.BERT_VOCAB= os.path.join(bert_path,'vocab.txt')
        self.BERT_INIT_CHKPNT =  os.path.join(bert_path,'bert_model.ckpt')
        self.BERT_CONFIG =  os.path.join(bert_path,'bert_config.json')
        
        tokenization.validate_case_matches_checkpoint(True, self.BERT_INIT_CHKPNT)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.BERT_VOCAB, do_lower_case=True)
        
        self.ID = 'guid'
        self.DATA_COLUMN = 'txt'
        self.LABEL_COLUMNS = ['Safety','CleanlinessView','Information','Service','Comfort','PersonnelCard','Additional']
        
        self.MAX_SEQ_LENGTH = 128
        
        # Compute train and warmup steps from batch size
        # These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 2e-5
        self.NUM_TRAIN_EPOCHS = 1

        # Warmup is a period of time where hte learning rate 
        
        # is small and gradually increases--usually helps training.
        self.WARMUP_PROPORTION = 0.1
        
        # Model configs
        self.SAVE_CHECKPOINTS_STEPS = 1000
        self.SAVE_SUMMARY_STEPS = 500
        
        self.run_config = tf.estimator.RunConfig(
            model_dir=output_path,
            save_summary_steps=self.SAVE_SUMMARY_STEPS,
            keep_checkpoint_max=1,
            save_checkpoints_steps=self.SAVE_CHECKPOINTS_STEPS)
        
        self.train_file = os.path.join(output_path, "train.tf_record")
        if not os.path.exists(self.train_file):
            open(self.train_file, 'w', encoding='utf8').close()
        
        self.eval_file = os.path.join(output_path, "eval.tf_record")
        if not os.path.exists(self.eval_file):
            open(self.eval_file, 'w', encoding='utf8').close()
            
        self.output_eval_file = os.path.join(output_path, "eval_results.txt")
        
    def create_examples(self, df, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[1]
            if labels_available:
                labels = row[2:]
            else:
                labels = [0,0,0,0,0,0,0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples
    
    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of `InputBatch`s."""
        
        features = []
        for (ex_index, example) in enumerate(examples):
            # print(example.text_a)
            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, self.MAX_SEQ_LENGTH - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.MAX_SEQ_LENGTH  - 2:
                    tokens_a = tokens_a[:(self.MAX_SEQ_LENGTH  - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (self.MAX_SEQ_LENGTH  - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.MAX_SEQ_LENGTH 
            assert len(input_mask) == self.MAX_SEQ_LENGTH 
            assert len(segment_ids) == self.MAX_SEQ_LENGTH 

            labels_ids = []
            for label in example.labels:
                labels_ids.append(int(label))

            if ex_index < 0:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_ids=labels_ids))
        return features

    def create_model(self, bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.sigmoid(logits)#### multi-label case

            labels = tf.cast(labels, tf.float32)
            tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
            loss = tf.reduce_mean(per_example_loss)

            return (loss, per_example_loss, logits, probabilities)

    def model_fn_builder(self, bert_config, num_labels, init_checkpoint, learning_rate,
                         num_train_steps, num_warmup_steps, use_tpu,
                         use_one_hot_embeddings):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]
            is_real_example = None
            if "is_real_example" in features:
                 is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
            else:
                 is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            (total_loss, per_example_loss, logits, probabilities) = self.create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)

            tvars = tf.trainable_variables()
            initialized_variable_names = {}
            scaffold_fn = None
            if init_checkpoint:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
                if use_tpu:

                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                        return tf.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            tf.logging.info("**** Trainable Variables ****")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:

                train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold=scaffold_fn)
            elif mode == tf.estimator.ModeKeys.EVAL:

                def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                    logits_split = tf.split(probabilities, num_labels, axis=-1)
                    label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                    # metrics change to auc of every class
                    eval_dict = {}

                    for j, logits in enumerate(logits_split):
                        label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)

                        current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                        eval_dict[str(j)] = (current_auc, update_op_auc)


                    eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)

                    return eval_dict


                eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)

                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=eval_metrics,
                    scaffold=scaffold_fn)
            else:
                print("mode:", mode,"probabilities:", probabilities)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"probabilities": probabilities},
                    scaffold=scaffold_fn)
            return output_spec

        return model_fn

    def input_fn_builder(self, features, seq_length, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_label_ids = []

        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "segment_ids":
                    tf.constant(
                        all_segment_ids,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "label_ids":
                    tf.constant(all_label_ids, shape=[num_examples, len(self.LABEL_COLUMNS)], dtype=tf.int32),
            })

            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                
            d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            return d

        return input_fn
    
    def convert_single_example(self, ex_index, example):
        """Converts a single `InputExample` into a single `InputFeatures`."""

        if isinstance(example, PaddingInputExample):
            return InputFeatures(
                input_ids=[0] * self.MAX_SEQ_LENGTH,
                input_mask=[0] * self.MAX_SEQ_LENGTH,
                segment_ids=[0] * self.MAX_SEQ_LENGTH,
                label_ids=0,
                is_real_example=False)

        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self.MAX_SEQ_LENGTH - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.MAX_SEQ_LENGTH - 2:
                tokens_a = tokens_a[0:(self.MAX_SEQ_LENGTH - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.MAX_SEQ_LENGTH:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == self.MAX_SEQ_LENGTH
        assert len(input_mask) == self.MAX_SEQ_LENGTH
        assert len(segment_ids) == self.MAX_SEQ_LENGTH

        labels_ids = []
        for label in example.labels:
            labels_ids.append(int(label))


        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=labels_ids,
            is_real_example=True)
        return feature


    def file_based_convert_examples_to_features(self, examples, output_file):
        """Convert a set of `InputExample`s to a TFRecord file."""
        
        
        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_index, example) in enumerate(examples):
            #if ex_index % 10000 == 0:
                #tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

            feature = self.convert_single_example(ex_index, example)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])
            if isinstance(feature.label_ids, list):
                label_ids = feature.label_ids
            else:
                label_ids = feature.label_ids[0]
            features["label_ids"] = create_int_feature(label_ids)

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        writer.close()


    def file_based_input_fn_builder(self, input_file, is_training, drop_remainder):
        """Creates an `input_fn` closure to be passed to TPUEstimator."""
        seq_length = self.MAX_SEQ_LENGTH
        
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([7], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
        }

        def _decode_record(record, name_to_features):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t

            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            # For training, we want a lot of parallel reading and shuffling.
            # For eval, we want no shuffling and parallel reading doesn't matter.
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)

            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record, name_to_features),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))

            return d

        return input_fn


    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
                
    def bert_warm_up(self, examples):
        self.num_train_steps = int(len(examples) / self.BATCH_SIZE * self.NUM_TRAIN_EPOCHS)
        self.num_warmup_steps = int(self.num_train_steps * self.WARMUP_PROPORTION)
        self.file_based_convert_examples_to_features(examples, output_file = self.train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(examples))
        tf.logging.info("  Batch size = %d", self.BATCH_SIZE)
        tf.logging.info("  Num steps = %d", self.num_train_steps)
    
    def generate_estimator(self):
        self.bert_config = modeling.BertConfig.from_json_file(self.BERT_CONFIG)
        self.model_fn = self.model_fn_builder(
          bert_config=self.bert_config,
          num_labels= len(self.LABEL_COLUMNS),
          init_checkpoint=self.BERT_INIT_CHKPNT,
          learning_rate=self.LEARNING_RATE,
          num_train_steps=self.num_train_steps,
          num_warmup_steps=self.num_warmup_steps,
          use_tpu=False,
          use_one_hot_embeddings=False)

        self.estimator = tf.estimator.Estimator(
          model_fn=self.model_fn,
          config=self.run_config,
          params={"batch_size": self.BATCH_SIZE})
    
    def train_bert(self, x_train_resampled):
        examples = self.create_examples(x_train_resampled)
        features = self.convert_examples_to_features(examples)
        self.bert_warm_up(examples)
        train_input_fn = self.file_based_input_fn_builder(input_file=self.train_file, is_training=True, drop_remainder=True)
        self.generate_estimator()
        
        print(f'Beginning Training!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        print("Training took time ", datetime.now() - current_time)
    
    def create_output(self, predictions):
        eps = 0.5
        outcome = []
        for (_, prediction) in enumerate(predictions):
            preds = prediction["probabilities"]
            out = [1 if x >= eps else 0 for x in preds]
            outcome.append(out)
        dff = pd.DataFrame(outcome)
        dff.columns = self.LABEL_COLUMNS
        return dff
    
    def testing_model(self, test_data):
        eval_examples = self.create_examples(test_data)
        self.file_based_convert_examples_to_features(
            eval_examples, self.eval_file)
        
        eval_features = self.convert_examples_to_features(eval_examples)
        
        print('Beginning Predictions!')
        current_time = datetime.now()
        eval_input_fn = self.input_fn_builder(features=eval_features, seq_length=self.MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
        eval_prediction = self.estimator.predict(eval_input_fn)
        print("Prediction took time ", datetime.now() - current_time)
        
        eval_predictedlabel = self.create_output(eval_prediction)
        y_val = test_data.drop(['guid','txt'], axis=1)
        
        report = classification_report(y_val, eval_predictedlabel)
        print("Classification report: \n", report)
        print("F1 micro averaging:",(f1_score(y_val, eval_predictedlabel, average='micro')))
        print ("ROC: ",(roc_auc_score(y_val, eval_predictedlabel)))
        
        return eval_predictedlabel, report, f1_score, roc_auc_score