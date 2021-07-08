'''
THE MAIN BODY OF TAGGER MODEL
'''
import tensorflow as tf
from bert_master import modeling
import os

tf.app.flags.DEFINE_string('bert_file', 'D:/pycharm/py_proj/General_tagging_toolbox/bert_master'
                                               '/chinese_L-12_H-768_A-12/', 'pretrained bert ckpt path')
tf.app.flags.DEFINE_bool('reuse_model', False, 'reload model from pre-trained version or fine-tuned version' )
FLAGS = tf.app.flags.FLAGS

class BilstmModel(object):
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode

    def add_placeholder(self):
        with tf.variable_scope("inputs"):
            self.x_inputs = tf.placeholder(tf.int32, [None, self.params.max_seq_len], "x_inputs")
            self.y_inputs = tf.placeholder(tf.int32, [None, self.params.max_seq_len], "y_inputs")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

    def load_embedding_matrix(self):
        with tf.variable_scope("embedding"):
            self.embedding = tf.get_variable("embedding",
                                             [self.params.vocab_size, self.params.embeded_size],
                                             tf.float32)
            # x_inputs.shape = [batch_size, max_seq_len]  ->  inputs.shape = [batch_size, max_seq_len, embedding_size]
            embedded_inputs = tf.nn.embedding_lookup(self.embedding, self.x_inputs)
            # The input sentence is still padding filled data.
            # Calculate the actual length of each sentence, that is, the actual length of the non-zero non-padding portion.
            length = tf.reduce_sum(tf.sign(self.x_inputs), 1)
            length = tf.cast(length, tf.int32)
        return embedded_inputs, length

    def lstm_cell(self, hidden_size, keep_prob):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
        return tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    def lstm_encoder(self, embedded_inputs, length, hidden_size, keep_prob):
        '''
        :param embedded_inputs: in Lstm-CRF version: word embedding; in Bert-Lstm-CRF version: Bert encoded output
        :param length: input sequence length
        :param hidden_size: ~
        :param keep_prob: ~
        :return:
        '''
        with tf.variable_scope("encode"):
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_cell(hidden_size, keep_prob),
                                                                        self.lstm_cell(hidden_size, keep_prob),
                                                                        embedded_inputs,
                                                                        sequence_length=length, dtype=tf.float32)

            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.reshape(output, [-1, self.params.hidden_size * 2])
        return output

    def crf_layer(self, score, length):
        with tf.variable_scope('crf_loss'):
            transition_params = tf.get_variable("trans_params", shape=[self.params.class_num, self.params.class_num],
                                                initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            if self.mode == 'test':
                return None, transition_params
            else:
                log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(score,
                                                                                       self.y_inputs,
                                                                                       length)
                loss = tf.reduce_mean(-log_likelihood)
                return loss, transition_params

    def project_input(self, embeded_input, shape, name=None):
        '''
        This function is used to adjust the last dimension of encoded_output(lstm encoded output or bert-lstm encoded output)
        to meet the requirement of crf_layer.
        :param embeded_input: the encoded_output of lstm or bert-lstm or bert(without lstm)
        :param shape: the shape of adjustment weight matrix
        :param name: each model has its own weight matrix
        :return:
        '''
        with tf.variable_scope('projection_op' if not name else name):
            W = tf.get_variable("W", shape=shape,
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            b = tf.get_variable("b", shape=[shape[1]], dtype=tf.float32,
                                initializer=tf.zeros_initializer())
            pred = tf.nn.xw_plus_b(embeded_input, W, b)
        return pred

    def build_graph(self):
        '''
        LSTM-CRF model
        '''
        print("model graph is building!")
        with tf.variable_scope('Bilstm_tag_model'):
            self.add_placeholder()
            embedded_inputs, self.length = self.load_embedding_matrix()
            bilstm_outputs = self.lstm_encoder(embedded_inputs, self.length, self.params.hidden_size,
                                               self.params.keep_prob)

            y_pred = self.project_input(bilstm_outputs,
                                        [self.params.hidden_size * 2, self.params.class_num])
            self.score = tf.reshape(y_pred, shape=[-1, self.params.max_seq_len, self.params.class_num])

            self.loss, self.transition_params = self.crf_layer(self.score, self.length)

        tf.summary.scalar('loss', self.loss)

        optimizar = tf.train.AdamOptimizer(learning_rate=self.params.lr)
        self.train_op = optimizar.minimize(self.loss, global_step=self.global_step)
        self.summaries = tf.summary.merge_all()

class BertLstmModel(BilstmModel):
    def __init__(self, params, mode=None, with_Lstm=True):
        super(BertLstmModel, self).__init__(params, mode)
        self.with_lstm = with_Lstm

    def load_bert(self, input_ids, input_mask):
        self.bert_config = modeling.BertConfig.from_json_file(os.path.join(FLAGS.bert_file, "bert_config.json"))
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.mode,
            input_ids=input_ids,
            input_mask=input_mask)
        # when the first time training, we need load the pre-trained bert,
        # after this, we need reload the fine-tuned model ckpt (reuse version)
        if not FLAGS.reuse_model:
            init_checkpoint = os.path.join(FLAGS.bert_file, "bert_model.ckpt")
            tvars = tf.trainable_variables()
            # load pre-trained BERT model
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        return model

    def build_graph(self):
        '''
        BERT-CRF OR BERT-LSTM-CRF (with lstm) model
        '''
        print("model graph is building!")
        with tf.variable_scope('Bert-based_tag_model'):
            self.add_placeholder()
            input_mask = tf.sign(self.x_inputs)
            length = tf.reduce_sum(input_mask, 1)
            length = tf.cast(length, tf.int32)
            bert = self.load_bert(self.x_inputs, input_mask)
            embedded_input = bert.get_sequence_output()  # embedded input: [batch_size, seq_length, embedding_size]
            if self.with_lstm:
                # bert-lstm-crf model
                embedded_input = self.lstm_encoder(embedded_input, length, self.bert_config.hidden_size,
                                                   self.params.keep_prob)  # !! the parameter difference between bert_config and Bilstm
                pred = self.project_input(embedded_input, [self.bert_config.hidden_size*2, self.params.class_num],
                                   name='with_lstm_proj')
            else:
                # bert-crf model
                pred = self.project_input(embedded_input, [self.bert_config.hidden_size, self.params.class_num],
                                          name='without_lstm_proj')

            self.score = tf.reshape(pred, shape=[-1, self.params.max_seq_len, self.params.class_num])

            self.loss, self.transition_params = self.crf_layer(self.score, length)

        tf.summary.scalar('loss', self.loss)

        optimizar = tf.train.AdamOptimizer(learning_rate=self.params.lr)
        self.train_op = optimizar.minimize(self.loss, global_step=self.global_step)
        self.summaries = tf.summary.merge_all()
