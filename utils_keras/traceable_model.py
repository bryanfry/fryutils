import os
import pickle
from datetime import datetime
import pandas as pd
from keras.models import Sequential
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def _get_filename(name, starttime, ext):
    """
    Function to build filenames for model, weights, or results
    from the model name and traiing start time.  Used to produce
    filenames for .json (model), .h5 (weights), and .pkl (results).
    """
    ext = ''.join([i for i in ext if i != '.'])
    return name + starttime.strftime('_%y%m%d_%H%M%S.') + ext


def _apply_plot_cosmetics(ax, x_label=None, y_label=None, title=None,
                         show_legend=True):
    """
    Given plt.Axes object ax, apply titles / cosmetics.
    """
    if x_label:
        ax.set_xlabel(x_label, fontsize=18)
    if y_label:
        ax.set_ylabel(y_label, fontsize=18)
    if title:
        ax.set_title(title, fontsize=22)
    if show_legend:
        ax.legend(fontsize=16, frameon=True, framealpha=0.8, edgecolor='gray')
    ax.tick_params(axis='both', labelsize=14, pad=10)
    return ax


class TimingCallback(Callback):
    """
    Keras Callback child class to record the training time for each
    epoch"""

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = datetime.now()

    def on_epoch_end(self, epoch, logs={}):
        logs['epoch_time_sec'] = (datetime.now() -
                                  self.starttime).total_seconds()


class TraceableModel(Sequential):
    """
    This class for a generic NN inherits from the basic Sequential
    neural net model of Keras.  It adds some features to aid with
    'traceability' of training... the idea here is to be able to train
    on AWS and save some important highlights of the training to a file
    that can be analyzed offline.

    High-level function train() ingests X and y data and performs the
    following tasks:

    - Single x/y Train / test split, stratified.
    - Optional one-hot encoding on y for Keras ingestion.
    - Fits the model
        - Records training and testing time for each epoch.
    - Evaluates the model after all training
        - one-hot decodes the output, if needed.
    - Returns a TraceableResult object (defined below) that
      encapsulates timing, eval predictions, and model training
      history.  The TraceableResult can pickle itself with
      auto-generated descriptive filename."""

    def __init__(self, layers=None, name=None):
        super().__init__(layers=layers, name=name)
        self.training_count = 0  # incremented with each call to train()

    @classmethod
    def one_hot_encode(cls, y):
        """
        One-hot encode a label sequence.

        Ingest 1-d sequence of labels.  Return a 2d one-hot-encoded
        matrix, along with columns labels for the matrix
        :param y: 1-d sequence of labels (string)
        :return: 2d one-hot encoded matrix and col names
        """
        if y is None:
            return None, None
        if len(y) < 1:
            return None, None
        df_ohe = pd.get_dummies(y)
        return df_ohe.values, df_ohe.columns.tolist()

    @classmethod
    def one_hot_decode(cls, y_ohe, labels):
        """
        Inverse of function one_hot_encode: convert 2d one-hot
        encoded matrix and column lables to 1d label sequence.
        :param y_ohe: 2d one-hot-encoded matrix.
        :param labels: sequence of column labels
        :return: 1d label list
        """
        print(labels)
        print(y_ohe.shape)
        if y_ohe is None or labels is None:
            return None
        if len(y_ohe) < 1:
            return None
        return [labels[i] for i in y_ohe.argmax(axis=1).tolist()]

    @classmethod
    def label_decode(cls, y_pred, labels):
        """
        Given list of integer classifier predictions (from
        predict_classes()), return the labels.  Just a simple lookup
        into list of labels.
        """
        return [labels[i] for i in y_pred]


    def train(self, X, y, batch_size, epochs, test_frac=0.5, random_state=42,
              is_classification=True, perform_one_hot_encoding=True,
              record_X_train=False, record_y_train=False, record_X_test=False,
              record_y_test = False, record_y_pred = False,
              additional_callbacks=[]):
        """
        Wraps train/test split, one-hot-encoding for classification,
        and fit() into one function.  Returns a TraceableResult object
        to examine timing, accuracy, and loss over epochs.

        :param X: features
        :param y: results
        :param batch_size: training batch size
        :param epochs: # epochs for training
        :param test_frac: float (0-1), fraction of X and y to go to
        evaluation data.  Balance will go to training.
        :param random_state: random seed for train/test split
        :param is_classification: Bool, must be true as of 3/10/18
        :param perform_one_hot_encoding: Bool
        :param record_X_train: Bool - save training X in result?
        :param record_y_train: Bool - save training y in result?
        :param record_X_test: Bool - save eval X in result?
        :param record_y_test: Bool - save eval y in result?
        :param record_y_pred: Bool - save final test output result?
        :param additional_callbacks: List, any additional keras
        Callbacks to call during training.
        :return: a TraceableResult object
        """
        assert is_classification, 'is_classification must be True as of 3/10/18.'

        y, labels_unique = self.one_hot_encode(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=test_frac, random_state=random_state,
                            stratify=y)
        self.starttime = datetime.now()
        train_history = self.fit(X_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_data=(X_test, y_test),
                    callbacks=[TimingCallback()] + additional_callbacks)
        endtime = datetime.now()
        duration = endtime - self.starttime
        self.training_count = self.training_count + 1
        final_eval_score, final_eval_acc = self.evaluate(X_test, y_test,
                                                         batch_size)
        if record_y_pred:
            if is_classification:
                y_pred = self.predict_classes(X_test)
                y_pred = self.label_decode(y_pred, labels_unique)
            else:
                pass # TOOD 3/6/18 -- add support for regression.


        # If X and Y values are not to be saved in the TraceableResult
        # object, we discard them here.
        if not record_X_train:
            X_train = None
        if not record_y_train:
            y_train = None
        if not record_X_test:
            X_test = None
        if not record_y_test:
            y_test = None
        if not record_X_train:
            X_train = None

        if perform_one_hot_encoding:
            y_train = self.one_hot_decode(y_train, labels_unique)
            y_test = self.one_hot_decode(y_test, labels_unique)

        r = TraceableResult(train_history, batch_size, test_frac, random_state,
                            self.training_count, final_eval_score,
                            final_eval_acc, X_train, y_train, X_test,
                            y_test, y_pred, self.name, self.starttime, duration)
        return r


    def save_model_and_weights(self, dir, fn_prefix=None, verbose=True):
        """"""

        #https://machinelearningmastery.com/save-load-keras-deep-learning-models/
        assert self.training_count > 0, """TraceableModel.save_model_and_weights() requires that train() has been 
        previously called on the model"""

        if fn_prefix is None:
            fp_json = os.path.join(dir, _get_filename(self.name, self.starttime,
                                                      '.json'))
            fp_h5 = os.path.join(dir, _get_filename(self.name, self.starttime,
                                                    '.h5'))
        else:
            fp_json = os.path.join(dir, fn_prefix + '.json')
            fp_h5 = os.path.join(dir, fn_prefix + '.h5')

        with open(fp_json, 'w') as f:
            f.write(self.to_json())
        if verbose:
            print('\nModel JSON written to ' + fp_json)
        self.save_weights(fp_h5)
        if verbose:
            print('\nModel weights written to ' + fp_h5)


class TraceableResult:

    """
    Container to hold the results of a train() call on a TraceableModel
    object.
    """
    def __init__(self, train_history, batch_size, test_frac, random_state,
                 training_count, final_eval_score, final_eval_acc, X_train, y_train,
                 X_test, y_test, y_pred, name, starttime, duration):

        self.history = train_history.history
        self.params = train_history.params
        self.epoch = train_history.epoch
        self.n_epochs = len(self.epoch)
        self.batch_size = batch_size
        self.test_frac = test_frac
        self.random_state = random_state
        self.training_count = training_count
        self.final_eval_score = final_eval_score
        self.final_eval_acc = final_eval_acc
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.name = name
        self.starttime = starttime
        self.duration = duration


    def save(self, dir, fn=None, verbose=True):
        """
        This method directs the TraceableResult to pickle itself.  The
        *.pkl is saved in the directory <dir>.  If filename f is not
        given, a filename will be constructed from the model name and
        training start time.

        :param dir: directory in which to save pickled TraceableResult
        :param fn: str, filename.  If None, filename will be
                   autogenerated from model name and training start time.
        :param verbose: bool
        """
        if not fn:
            fn = _get_filename(self.name, self.starttime, '.pkl')
        fp = os.path.join(dir, fn)
        with open(fp, 'wb') as f:
            pickle.dump(self, f)
        if verbose:
            print('\nSaved TraceableResult to ' + fp)
        return fp

    def plot_results(self, show=True):
        """
        Generates matplotplots of training/eval accuracy, loss, and
        timings vs. epoch.
        :param show: Bool, show immediately.
        :return: plt.Figure and [plt.Axes]
        """
        fig, axes = plt.subplots(nrows=3, sharex=True)
        axes[0].plot(self.epoch, self.history['acc'], marker='o',
                     label='Training Accuracy')
        axes[0].plot(self.epoch, self.history['val_acc'], marker='o',
                     label='Testing Acuracy')
        _apply_plot_cosmetics(axes[0], y_label='Accuracy')
        axes[1].plot(self.epoch, self.history['loss'], marker='o',
                     label='Training Loss')
        axes[1].plot(self.epoch, self.history['val_loss'], marker='o',
                     label='Testing Loss')
        _apply_plot_cosmetics(axes[1], y_label='Loss')
        axes[2].plot(self.epoch, self.history['loss'], marker='o',
                     label='Training Time')
        axes[2].plot(self.epoch, self.history['epoch_time_sec'], marker='o',
                     label='Testing Time')
        _apply_plot_cosmetics(axes[2], x_label='Epoch', y_label='Elapsed Time[s]',
                             title=None)
        fig.set_figheight(11)
        fig.set_figwidth(8)
        if show:
            plt.show()
        return fig, axes


def load_result(fp):
    with open(fp, 'rb') as f:
        return pickle.load(f)