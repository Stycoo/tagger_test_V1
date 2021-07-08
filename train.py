import tensorflow as tf
import time
import numpy as np
from tqdm import trange
from data_loader import DataLoad
from model import BilstmModel, BertLstmModel
from params_config import params
# import os

tf.app.flags.DEFINE_string('data_path', '', 'train data path')
tf.app.flags.DEFINE_string('vocab_path', '', 'vocab file path')
tf.app.flags.DEFINE_string('model_save_path', '', 'trained model save path')
tf.app.flags.DEFINE_string('task', '', 'choose the task type (NER/POS)')
tf.app.flags.DEFINE_string('model_type', '', 'choose the model type (Bilstm/BertLstm)')
tf.app.flags.DEFINE_string('mode', '', 'train or test')
tf.app.flags.DEFINE_string('with_lstm', '', 'use lstm or not')

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

class ModelTrain(object):
    def __init__(self, train_data, valid_data, test_data, model=None):
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.model = model
        self.model_params = params()

    def train(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        train_batch_size = self.model_params.batch_size
        train_epoch = self.model_params.epoch
        train_batch_num = int(self.train_data.data_size / train_batch_size)

        display_num = 5  # display 5 loss pre epoch
        display_batch = int(train_batch_num / display_num)

        sess = tf.Session(config=config)
        saver = tf.train.Saver(max_to_keep=10)

        ckpt = tf.train.latest_checkpoint(FLAGS.model_save_path)
        if ckpt is None:
            tf.logging.info("Initializing model...")
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, ckpt)
        summary_writer = tf.summary.FileWriter(FLAGS.model_save_path, sess.graph)

        for epoch in range(train_epoch):
            tf.logging.info("Epoch: %d    lr: %f" % (epoch, self.model_params.lr))
            start_time = time.time()
            loss = 0
            display_loss = 0

            for batch_ind in trange(train_batch_num):
                fetches = [self.model.loss, self.model.train_op, self.model.summaries, self.model.global_step]
                x_inputs, y_inputs = self.train_data.next_batch(train_batch_size)
                feed_dict = {self.model.x_inputs: x_inputs, self.model.y_inputs: y_inputs}
                _loss, _, _summaries, _global_step = sess.run(fetches, feed_dict)

                summary_writer.add_summary(_summaries, _global_step)
                loss += _loss
                display_loss += _loss

                # early stop (unfinished)
                if batch_ind % display_batch == 0:
                    valid_acc = self.valid(self.valid_data, sess)
                    tf.logging.info("Train loss: %g Valid acc :%g" % (display_loss / display_batch, valid_acc))
                    display_loss = 0

            mean_loss = loss / train_batch_num
            saver.save(sess, FLAGS.model_save_path, global_step=(epoch + 1))
            tf.logging.info("Epoch training loss: %g  time cost: %g" % (mean_loss, time.time()-start_time))

    def valid(self, dataset, sess):
        dataset_size = dataset.shape[0]
        batch_num = int(dataset_size / self.model_params.test_batch_size)
        correct_labels_num = 0
        total_labels_num = 0
        fetches = [self.model.score, self.model.length, self.model.transition_params]
        for i in range(batch_num):
            x_inputs, y_inputs = dataset.next_batch(FLAGS.test_batch_size)
            feed_dict = {self.model.x_inputs: x_inputs, self.model.y_inputs: y_inputs}
            pre_score, pre_length, pre_transition_params = sess.run(fetches, feed_dict)

            for score, length, y_inp in zip(pre_score, pre_length, y_inputs):
                score = score[:length]
                y_inp = y_inp[:length]
                pre_sequence, _ = tf.contrib.crf.viterbi_decode(score, pre_transition_params)
                correct_labels_num += np.sum(np.equal(pre_sequence, y_inp))
                total_labels_num += length

        acc = correct_labels_num / total_labels_num
        return acc

def main():
    model_params = params()
    data = DataLoad(FLAGS.vocab_path, FLAGS.data_path)
    train_batcher, valid_batcher, test_batcher = data.buildTraindata()

    '''
    choose to train Bilstm-CRF or Bert-Lstm-CRF
    '''
    if FLAGS.model_type == 'Bilstm':
        model = BilstmModel(model_params, FLAGS.mode)
        model.build_graph()
    elif FLAGS.model_type == 'BertLstm':
        model = BertLstmModel(model_params, FLAGS.mode, FLAGS.with_lstm)
        model.build_graph()

    trainer = ModelTrain(train_batcher, valid_batcher, test_batcher, model)
    trainer.train()

if __name__ == "__main__":
    tf.app.run()