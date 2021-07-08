'''
TAGGING CLIENT
by setting parameter task, you can choose to do POS task or NER task;
by setting parameter model_type, you can choose to use LSTM-CRF or BERT-LSTM-CRF to carry out your task.
'''
import tensorflow as tf
from data_loader import DataLoad
from util import str2token, token2id
from model import BilstmModel, BertLstmModel
from params_config import params

tf.app.flags.DEFINE_string('vocab_path', '', 'vocab file path')
tf.app.flags.DEFINE_string('model_save_path', '', 'trained model save path')
tf.app.flags.DEFINE_string('task', '', 'choose the task type (NER/POS)')
tf.app.flags.DEFINE_string('model_type', '', 'choose the model type (Bilstm/BertLstm)')
tf.app.flags.DEFINE_string('mode', '', 'train or test')
tf.app.flags.DEFINE_string('with_lstm', '', 'use lstm or not')

FLAGS = tf.app.flags.FLAGS

class Tagger(object):
    def __init__(self):
        self.model_params = params()
        self.data = DataLoad(vocab_path=FLAGS.vocab_path)

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.sess.as_default():
            with self.graph.as_default():
                if FLAGS.model_type == 'Bilstm':
                    self.model = BilstmModel(self.model_params, FLAGS.mode)
                elif FLAGS.model_type == 'BertLstm':
                    self.model = BertLstmModel(self.model_params, FLAGS.mode, FLAGS.with_lstm)

                self.model.build_graph()
                ckpt = tf.train.latest_checkpoint(FLAGS.model_save_path)
                tf.train.Saver().restore(self.sess, ckpt)

    def predict(self, text) -> list:
        with self.sess.as_default():
            if text:
                x_input_str = str2token(text, task_type=FLAGS.task_type)
                x_input = token2id(x_input_str, self.data.word2id, self.model_params.max_seq_len)  # np.array
                fetches = [self.model.score, self.model.length, self.model.transition_params]
                feed_dict = {self.model.x_inputs: x_input}
                pre_scores, length, transition_params = self.sess.run(fetches, feed_dict)
                pre_tags, _ = tf.contrib.crf.viterbi_decode(pre_scores[0][:length[0]], transition_params)
                res_tags = [self.data.id2tag[i] for i in pre_tags]
                return res_tags
            else:
                print("please input again: ")
                return []

def main():
    tagger = Tagger()
    while True:
        input_text = input()
        if input_text == '\n':
            break
        tag_res = tagger.predict(input_text)
        if not tag_res:
            continue
        else:
            print(tag_res)

if __name__ == "__main__":
    tf.app.run()
