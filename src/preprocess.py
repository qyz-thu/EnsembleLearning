import numpy as np
import re
import argparse
Vocab = dict()


def get_vocab(save_path, remove_head, vocab_size):
    vocab = dict()
    with open('data/train.csv') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            data = line.strip().split('\t')
            words = re.compile('[a-z\-]+')
            summary = words.findall(data[4].lower())
            review = words.findall(data[5].lower())
            for word in summary:
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
            for word in review:
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
            pass
        print("total train sample: %d" % i)
    print("total word: %d" % len(vocab))
    index = 0
    freq_list = [[vocab[word], word] for word in vocab]
    freq_list.sort(key=lambda w: w[0], reverse=True)    # sort by frequency
    freq_list = freq_list[remove_head: remove_head + vocab_size]     # remove words with high frequency
    print(freq_list[-1])
    with open(save_path, 'w') as f:
        for word in freq_list:
            f.write(word[1] + '\t' + str(index) + '\n')
            index += 1
    print("write vocabulary %d" % index)


def process_train(summary_weight):
    global Vocab
    data = []
    label = []
    with open('data/train.csv') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            vector = [0 for i in range(len(Vocab))]
            tokens = line.strip().split('\t')
            label.append(float(tokens[0]))
            pattern = re.compile('[a-z\-]+')
            summary = pattern.findall(tokens[4].lower())
            review = pattern.findall(tokens[5].lower())
            for word in summary:
                if word in Vocab:
                    vector[Vocab[word]] += summary_weight
            for word in review:
                if word in Vocab:
                    vector[Vocab[word]] += 1
            data.append(vector)
    data = np.array(data)
    label = np.array(label)
    np.save('data/train.npy', data)
    np.save('data/label.npy', label)


def process_test(summary_weight):
    global Vocab
    data = []
    with open('data/test.csv') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.strip().split('\t')
            vector = [0 for i in range(len(Vocab))]
            pattern = re.compile('[a-z\-]+')
            summary = pattern.findall(tokens[4].lower())
            review = pattern.findall(tokens[5].lower())
            for word in summary:
                if word in Vocab:
                    vector[Vocab[word]] += summary_weight
            for word in review:
                if word in Vocab:
                    vector[Vocab[word]] += 1
            data.append(vector)
            pass
    data = np.array(data)
    np.save('data/test.npy', data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove_head", type=int, default=20)
    parser.add_argument("--vocab_size", type=int, default=2000)
    parser.add_argument("--summary_weight", type=int, default=5)
    args = parser.parse_args()

    get_vocab('data/vocab.txt', args.remove_head, args.vocab_size)
    with open('data/vocab.txt') as f:
        for line in f:
            tokens = line.strip().split('\t')
            Vocab[tokens[0]] = int(tokens[1])
    process_test(args.summary_weight)
    process_train(args.summary_weight)


if __name__ == "__main__":
    main()
