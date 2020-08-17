import numpy as np
import pandas as pd
import time


def load_data(which='SG'):
    if which == 'oppo_len64':
        train_x = np.load('Data/seglen64/train/data.npy')
        train_y = np.load('Data/seglen64/train/train_y_denull.npy')
        train_y = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
        test_x = np.load('Data/seglen64/test/data.npy')
        test_y = np.load('Data/seglen64/test/test_y_denull.npy')
        test_y = np.asarray(pd.get_dummies(test_y), dtype=np.int8)
        print('#  OPPO loaded.')
        return train_x, train_y, test_x, test_y
    elif which == 'oppo_len64_pkc':
        train_x = np.load('Data/oppo_len64/train_x_denull.npy')
        train_y = np.load('Data/oppo_len64/train_y_denull.npy')
        c_tr = 0
        for a in train_y:
            if a in [1, 2, 5, 7, 9, 11, 13, 15, 16, 17]:
                c_tr += 1
        new_train_x = np.zeros([c_tr, 64, 113])
        new_train_y = np.zeros([c_tr])
        new_train_y = np.asarray(pd.get_dummies(new_train_y), dtype=np.int8)
        test_x = np.load('Data/oppo_len64/test_x_denull.npy')
        test_y = np.load('Data/oppo_len64/test_y_denull.npy')
        test_y = np.asarray(pd.get_dummies(test_y), dtype=np.int8)
        print('#  OPPO loaded.')
        return new_train_x, new_train_y, test_x, test_y
    else:
        pass


def shuffle(data, labels):
    index = np.arange(len(data))
    np.random.shuffle(index)
    return data[index], labels[index]


def cross_val(data, labels, epoch, cross=5):
    l = len(data)
    i = epoch % cross
    j = np.ceil(l / cross)
    start_idx = int(i * j)
    end_idx = int((i + 1) * j)
    train_xc = np.append(data[:start_idx], data[end_idx:], axis=0)
    train_yc = np.append(labels[:start_idx], labels[end_idx:], axis=0)
    val_xc = data[start_idx:end_idx]
    val_yc = labels[start_idx:end_idx]
    return train_xc, train_yc, val_xc, val_yc


def extract_batch_size(_train, epoch, batch_size):
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)
    for i in range(batch_size):
        index = ((epoch - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]
    return batch_s


def sub_max(arr, n):
    arr_ = np.empty(shape=arr.shape, dtype=arr.dtype)
    _arr = np.array([], dtype=arr.dtype)
    _arr = np.append(_arr, arr)
    # print(_arr)
    if n <= 1:
        return np.max(_arr), np.argmax(_arr)
    else:
        for i in range(n - 1):
            arr_ = _arr
            arr_[np.argmax(arr_)] = np.min(_arr)
            _arr = arr_
        return np.max(arr_), np.argmax(arr_)


def load_captions_data(path='Data/oppo_len300denull/', split='train/'):
    data_path = path + split
    dt = np.load(data_path + 'data.npy')
    # dt = np.expand_dims(dt, axis=-1)
    data = {'data': dt,
            'captions': np.load(data_path + 'caps.npy'),
            # 'captions': np.load(data_path + 'captions_with_null.npy'),
            # 'captions_vector': np.load(data_path + 'captions_vector.npy'),
            'word_to_idx': {'<null>': 0,
                            '<Open_Door_1>': 1,
                            '<Open_Door_2>': 2,
                            '<Close_Door_1>': 3,
                            '<Close_Door_2>': 4,
                            '<Open_Fridge>': 5,
                            '<Close_Fridge>': 6,
                            '<Open_Dishwasher>': 7,
                            '<Close_Dishwasher>': 8,
                            '<Open_Drawer 1>': 9,
                            '<Close_Drawer 1>': 10,
                            '<Open_Drawer 2>': 11,
                            '<Close_Drawer 2>': 12,
                            '<Open_Drawer 3>': 13,
                            '<Close_Drawer 3>': 14,
                            '<Clean_Table>': 15,
                            '<Drink_from Cup>': 16,
                            '<Toggle_Switch>': 17,
                            '<start>': 18, '<end>': 19},
            'data_idxs': np.arange(0, np.load(data_path + 'caps.npy').shape[0], dtype=int)}
    # 'idx_to_word': {0: 'Gap', 1: 'A', 2: 'B', 3: 'C', 4: '<start>', 5: '<end>'}
    print('load {} finished'.format(data_path))
    return data


def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        T = captions.shape[0]
        N = 1
    else:
        N, T = captions.shape

    decoded = []
    for i in range(N):
        words = []
        for t in range(T):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<end>':
                words.append('<end>')
                break
            if word != '<null>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def word_vector(inputs, n_words):  # one-hot
    dig = np.diag([1] * n_words)
    labels = inputs
    cap_vector = np.empty(shape=(0, labels.shape[1], n_words))
    for n in labels:
        words_vector = np.empty(shape=(0, n_words))
        for m in n:
            m = int(m)
            # print(m)
            words_vector = np.vstack((words_vector, dig[m]))
            # print(words_vector.shape)
        cap_vector = np.vstack((cap_vector, words_vector.reshape(1, labels.shape[1], n_words)))
    # print(cap_vector.shape)
    return cap_vector


def pick_data(dataset, labels, pick_labels, save_path):
    dataset_ = np.empty([0, dataset.shape[1], dataset.shape[2]])
    labels_ = np.empty(0)
    for i in range(dataset.shape[0]):
        if labels[i] in pick_labels:
            # print(labels[i])
            dataset_ = np.vstack([dataset_, dataset[i:i + 1]])
            labels_ = np.append(labels_, labels[i])
        process_bar(i, dataset.shape[0], 20)
    np.save(save_path + "/data.npy", dataset_)
    np.save(save_path + '/labels_onehot.npy', np.asarray(pd.get_dummies(labels_), dtype=np.int8))
    np.save(save_path + '/labels.npy', labels_)
    print('Pick Finished', pick_labels)


def process_bar(current_state, total_state, bar_length=20):
    current_bar = int(current_state / total_state * bar_length)
    bar = ['['] + ['#'] * current_bar + ['-'] * (bar_length - current_bar) + [']']
    bar_show = ''.join(bar)
    print('\r{}%d%%'.format(bar_show) % ((current_state + 1) / total_state * 100), end='')
    if current_state == total_state - 1:
        bar = ['['] + ['#'] * bar_length + [']']
        bar_show = ''.join(bar)
        print('\r{}%d%%'.format(bar_show) % 100, end='')
        print('\r')
