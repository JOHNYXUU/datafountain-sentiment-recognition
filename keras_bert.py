import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
from keras.utils import to_categorical
import os
import six
from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
from transformers import *
import transformers


def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""

    def return_id(str1, truncation_strategy, length):
        inputs = tokenizer.encode_plus(str1,
                                       add_special_tokens=True,
                                       max_length=length,
                                       truncation_strategy=truncation_strategy)

        input_ids = inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)

        return [input_ids, input_masks, input_segments]

    input_ids, input_masks, input_segments = return_id(
        instance, 'longest_first', max_sequence_length)

    return [input_ids, input_masks, input_segments]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[columns]):
        ids, masks, segments = \
            _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)

        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32),
            np.asarray(input_masks, dtype=np.int32),
            np.asarray(input_segments, dtype=np.int32)
            ]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns].astype(int) + 1)
outputs = compute_output_arrays(df_train, output_categories)

def create_model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)
    config = BertConfig.from_pretrained('bert-base-chinese')
    bert_model = TFBertModel.from_pretrained('bert-base-chinese',config=config)
    embedding = bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]
    x = tf.keras.layers.GlobalAveragePooling1D()(embedding)
    x = tf.keras.layers.Dropout(0.2)(x)
    x1 = tf.keras.layers.Dense(3, activation='softmax',name='class_out')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x1)
    return model

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = [1.,1.,1.]
    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        X_p = np.argmax(X_p*coef,axis=1)
        y_t = np.argmax(y,axis=1)
        ll = f1_score(y_t, X_p,average='macro')
        print('f1 score: ',ll)
        return -ll
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        if type(self.coef_) is list:
          initial_coef = self.coef_
        else:
          initial_coef = self.coef_['x']
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef,method='Nelder-Mead')
    def predict(self, X, coef):
        X_p = np.copy(X)
        X_p = np.argmax(X_p*coef,axis=1)
        return X_p
    def coefficients(self):
        return self.coef_['x']

def main():
    PATH ='/content/drive/My Drive/Colab Notebooks/data'
    MAX_SEQUENCE_LENGTH = 150
    input_categories = '微博中文内容'
    output_categories = '情感倾向'
    df_train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/nCoV_100k_train.labled.csv')
    df_train = df_train[df_train[output_categories].isin(['-1','0','1'])]
    df_test = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/nCov_10k_test.csv')
    df_sub = pd.read_csv('/content/drive/My Drive/Colab Notebooks/data/submit_example.csv')

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
    test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)

    gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X=df_train[input_categories].fillna('-1'),
                                                                           y=df_train[output_categories].fillna('-1'))
    valid_preds = []
    test_preds = []
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = to_categorical(outputs[train_idx])

        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = to_categorical(outputs[valid_idx])
        K.clear_session()
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model = create_model()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
        model.summary()
        model.fit(train_inputs, train_outputs, validation_data=[valid_inputs, valid_outputs], epochs=1, batch_size=64)
        valid_preds.append(model.predict(valid_inputs))
        test_preds.append(model.predict(test_inputs))


    opr = OptimizedRounder()


    gkf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0).split(X=df_train[input_categories].fillna('-1'), y=df_train[output_categories].fillna('-1'))
    for fold, (train_idx, valid_idx) in enumerate(gkf):
      print('flod: ',fold)
      valid_outputs = to_categorical(outputs[valid_idx])
      opr.fit(X=valid_preds[fold],y=valid_outputs)

    coef = opr.coef_['x']
    sub = np.average(test_preds, axis=0)
    sub = opr.predict(sub, coef)
    df_sub['y'] = sub-1
    df_sub.astype('int64').to_csv('test_sub.csv',index=False, encoding='utf-8')

main()
