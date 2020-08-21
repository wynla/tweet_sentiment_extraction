import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
print('TF version',tf.__version__)

def loss_fn(y_true, y_pred):
    # adjust the targets for sequence bucketing
    ll = tf.shape(y_pred)[1]
    y_true = y_true[:, :ll]
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,
        from_logits=False, label_smoothing=0.1)
    loss = tf.reduce_mean(loss)
    return loss

#模型建立
def build_model():
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    config = RobertaConfig.from_pretrained(PATH + 'config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained(PATH + 'pretrained-roberta-base.h5', config=config)
    x = bert_model(ids, attention_mask=att, token_type_ids=tok)

    x1 = tf.keras.layers.Dropout(0.1)(x[0])
    x1 = tf.keras.layers.Conv1D(768, 2, padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)

    x2 = tf.keras.layers.Dropout(0.1)(x[0])
    x2 = tf.keras.layers.Conv1D(768, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1, x2])
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss=loss_fn, optimizer=optimizer)

    return model


MAX_LEN = 96
PATH = 'robert/'
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file=PATH+'vocab-roberta-base.json',
    merges_file=PATH+'merges-roberta-base.txt',
    lowercase=True,
    add_prefix_space=True
)
SEED = 88888
LABEL_SMOOTHING = 0.1
tf.random.set_seed(SEED)
np.random.seed(SEED)
sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
train = pd.read_csv('input/extended_train.csv').fillna('')

# ct = train.shape[0]
# input_ids = np.ones((ct, MAX_LEN), dtype='int32')
# attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
# token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
# start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
# end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
#
# for k in range(train.shape[0]):
#     # FIND OVERLAP
#     text1 = " " + " ".join(train.loc[k, 'text'].split())
#     text2 = " ".join(train.loc[k, 'selected_text'].split())
#     idx = text1.find(text2)
#     chars = np.zeros((len(text1)))
#     chars[idx:idx + len(text2)] = 1
#     if text1[idx - 1] == ' ': chars[idx - 1] = 1
#     enc = tokenizer.encode(text1)
#     # ID_OFFSETS
#     offsets = [];
#     idx = 0
#     for t in enc.ids:
#         w = tokenizer.decode([t])
#         offsets.append((idx, idx + len(w)))
#         idx += len(w)
#     # START END TOKENS
#     toks = []
#     for i, (a, b) in enumerate(offsets):
#         sm = np.sum(chars[a:b])
#         if sm > 0: toks.append(i)
#     s_tok = sentiment_id[train.loc[k, 'sentiment']]
#     input_ids[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
#     attention_mask[k, :len(enc.ids) + 5] = 1
#     if len(toks) > 0:
#         start_tokens[k, toks[0] + 1] = 1
#         end_tokens[k, toks[-1] + 1] = 1

test = pd.read_csv('input/test.csv').fillna('')

ct = test.shape[0]
input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')


preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))
DISPLAY = 1
for i in range(1):
    print('#' * 25)
    print('### MODEL %i' % (i + 1))
    print('#' * 25)

    K.clear_session()
    model = build_model()
    model.load_weights('pretrain/output/'+'v4-roberta-%i.h5' % i)

    print('Predicting Test...')
    preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)
    preds_start += preds[0] / 5
    preds_end += preds[1] / 5

all = []
for k in range(input_ids_t.shape[0]):
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b:
        st = test.loc[k,'text']
    else:
        text1 = " "+" ".join(test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)
print(all)

test['selected_text'] = all
test[['textID','selected_text']].to_csv('submission.csv',index=False)