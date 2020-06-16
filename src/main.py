import numpy as np
from sklearn import svm, tree
from model import MLPClassifier
import torch
from torch import nn, optim
import time
import argparse


def train(data, label, args, bagging=True, use_svm=True):
    print("start training")
    print("use svm" if use_svm else "use dt")
    print("use bagging" if bagging else "use adaboost")
    start_time = time.time()
    classifier = []
    betas = None
    if bagging:
        for i in range(args.bagging_iter):
            print("iter %d" % i)
            size = len(data)
            samples = np.random.choice(size, size)
            sampled_data = np.array([data[samples[j]] for j in range(size)])
            sampled_label = np.array([label[samples[j]] for j in range(size)])
            # choose classifier
            clf = svm.LinearSVC(max_iter=args.max_iter) if use_svm else tree.DecisionTreeClassifier(max_depth=args.max_depth)
            clf.fit(sampled_data, sampled_label)
            classifier.append(clf)
            print("time used %.2f" % (time.time() - start_time))
    else:   # adaboost M1
        betas = []
        weights = np.ones(len(data))
        for i in range(args.adaboost_iter):
            print("iter %d" % i)
            # choose classifier
            clf = svm.LinearSVC(max_iter=args.max_iter) if use_svm else tree.DecisionTreeClassifier(max_depth=args.max_depth)
            clf.fit(data, label, sample_weight=weights)
            epsilon = 1 - clf.score(data, label)
            if epsilon > 0.5:
                print("hypothesis too weak!")
                print("skip")
                break
            beta = epsilon / (1 - epsilon)
            predict = clf.predict(data)
            for j in range(predict.shape[0]):
                if predict[j] == label[j]:
                    weights[j] *= beta
            weights *= len(data) / np.sum(weights)      # normalize
            betas.append(beta)
            classifier.append(clf)
            print("time used %.2f" % (time.time() - start_time))
        assert len(classifier) == len(betas)

    print("done training")
    print("training time %.2f" % (time.time() - start_time))
    return classifier, betas


def test(data, label, classifier, betas):
    res_sum = np.zeros(len(data))
    weights = [1 for i in range(len(classifier))] if not betas else [-np.log(betas[i]) for i in range(len(betas))]
    for i, clf in enumerate(classifier):
        result = clf.predict(data)
        res_sum += result * weights[i]
    Sum = sum(weights)
    res_sum /= Sum
    res_sum -= label
    rmse = np.linalg.norm(res_sum) / np.sqrt(len(data))
    print("RMSE: %.4f" % rmse)


def generate(data, classifier, betas):
    res_sum = np.zeros(len(data))
    weights = [1 for i in range(len(classifier))] if not betas else [-np.log(betas[i]) for i in range(len(betas))]
    for i, clf in enumerate(classifier):
        result = clf.predict(data)
        res_sum += result * weights[i]
    Sum = sum(weights)
    res_sum /= Sum
    with open('prediction.csv', 'w') as f:
        f.write("id,predicted\n")
        for i in range(len(res_sum)):
            f.write(str(i+1) + ',' + str(res_sum[i]) + '\n')


def train_mlp(data, label, iter=10, hidden_size=256, batch_size=10, lr=0.0001):
    models = []
    start_time = time.time()
    print("start training mlp")
    input_size = data.shape[1]
    for i in range(iter):
        print("mlp iter %d" % i)
        size = len(data)
        samples = np.random.choice(size, size)
        sampled_data = torch.tensor([data[samples[j]] for j in range(size)]).float()
        sampled_label = torch.tensor([label[samples[j]] for j in range(size)]).float()
        model = MLPClassifier(input_size, hidden_size)
        current_step = 0
        loss_function = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        step = 0
        while current_step < size:
            step += 1
            batch = sampled_data[current_step: current_step + batch_size]
            batch_label = sampled_label[current_step: current_step + batch_size]
            current_step += batch_size
            predict = model(batch)
            loss = loss_function(predict, batch_label)
            loss.backward()
            if (step + 1) % 1000 == 0:
                print(loss)
            optimizer.step()
            model.zero_grad()
        models.append(model)
        print("time used %.2f" % (time.time() - start_time))

    return models


def generate_mlp(data, models):
    res_sum = np.zeros(len(data))
    data = torch.tensor(data).float()
    for i, model in enumerate(models):
        result = model(data).detach().numpy()
        res_sum += result
    res_sum /= len(models)
    with open('prediction.csv', 'w') as f:
        f.write("id,predicted\n")
        for i in range(len(res_sum)):
            f.write(str(i+1) + ',' + str(res_sum[i]) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bagging_iter", type=int, default=1)
    parser.add_argument("--adaboost_iter", type=int, default=10)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--use_mlp", type=bool, default=False)
    parser.add_argument("--use_svm", type=bool, default=False)
    parser.add_argument("--ensemble_type", type=str, default="bagging")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--generate", type=bool, default=False)
    args = parser.parse_args()

    train_data = np.load('data/train.npy')
    label = np.load('data/label.npy')
    to_generate = args.generate
    if not to_generate:
        dev_data = train_data[210000:]
        dev_label = label[210000:]
        train_data = train_data[:210000]
        label = label[:210000]

    use_bagging = args.ensemble_type == 'bagging'

    if args.use_mlp:
        models = train_mlp(train_data, label, iter=args.bagging_iter, hidden_size=args.hidden_size, batch_size=args.batch_size)
        test_data = np.load('data/test.npy')
        generate_mlp(test_data, models)
    else:
        classifier, betas = train(train_data, label, args, bagging=use_bagging, use_svm=args.use_svm)
        if to_generate:
            test_data = np.load('data/test.npy')
            generate(test_data, classifier, betas)
        else:
            test(dev_data, dev_label, classifier, betas)


if __name__ == "__main__":
    main()
