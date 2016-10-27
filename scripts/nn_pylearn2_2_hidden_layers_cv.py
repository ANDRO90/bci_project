import theano
from pylearn2.models import mlp
from pylearn2.costs.mlp import WeightDecay, Default
from pylearn2.costs.cost import SumOfCosts
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from sklearn import cross_validation
import numpy as np
import h5py

TR = h5py.File('train.mat')
TE = h5py.File('test.mat')

pos_x_tr = np.array(TR['pos_x_tr']).T
pos_y_tr = np.array(TR['pos_y_tr']).T

feats_tr = np.load("features_tr.npy")
feats_te = np.load("features_te.npy")

X = feats_tr
y = np.c_[pos_x_tr, pos_y_tr]

n_folds = 10
batch_size = 100

inputs = 160
dims = [5, 10, 25, 32, 50]
dim2 = 10
hidden_types = ["sigmoid", "tanh"]
learning_rates = [.01, .02, .03, .05]
epochs = [15, 30, 50, 100, 200]

n_models = len(dims) * len(learning_rates) * len(epochs) * len(hidden_types)

scores = np.zeros(shape=(n_models, n_folds))
hyperparameters = list()

k_fold = cross_validation.KFold(len(X), n_folds)

for k, (train, test) in enumerate(k_fold):
    i = 0
    for epoch in epochs:
        for dim in dims:
            for learning_rate in learning_rates:
                for hidden_type in hidden_types:
                    if k == 0:
                        hyperparameters.append([dim, learning_rate, epoch, hidden_type])
                    print k, [dim, learning_rate, epoch, hidden_type]

                    ds = DenseDesignMatrix(X=X[train], y=y[train])

                    if hidden_type == "rectified":
                        hidden_layer1 = mlp.RectifiedLinear(layer_name='hidden1', dim=dim, irange=.1, init_bias=1.)
                    else:
                        if hidden_type == "tanh":
                            hidden_layer1 = mlp.Tanh(layer_name='hidden1', dim=dim, irange=.1, init_bias=1.)
                        else:
                            if hidden_type == "sigmoid":
                                hidden_layer1 = mlp.Sigmoid(layer_name='hidden1', dim=dim, irange=.1, init_bias=1.)

                    # statically cast second hidden layer to Tanh because it achieved a better performance
                    hidden_layer2 = mlp.Tanh(layer_name='hidden2', dim=dim2, irange=.1, init_bias=1.)
                    output_layer = mlp.Linear(dim=2, layer_name='y', irange=.1)

                    default = Default()
                    wdecay = WeightDecay(coeffs={'hidden1': 0.0001, 'hidden2': 0.0001, 'y': 0.0001})
                    costs = SumOfCosts([default, wdecay])

                    trainer = sgd.SGD(learning_rate=learning_rate, batch_size=batch_size,
                                      termination_criterion=EpochCounter(epoch), cost=costs)

                    layers = [hidden_layer1, hidden_layer2, output_layer]
                    ann = mlp.MLP(layers, nvis=inputs)
                    trainer.setup(ann, ds)

                    while True:
                        trainer.train(dataset=ds)
                        ann.monitor.report_epoch()
                        ann.monitor()

                        if not trainer.continue_learning(ann):
                            break

                    est = ann.fprop(theano.shared(X[test], name='inputs')).eval()
                    e_x = est[:, 0]
                    e_x = e_x.reshape(len(e_x), 1) - pos_x_tr[test]
                    e_y = est[:, 1]
                    e_y = e_y.reshape(len(e_y), 1) - pos_y_tr[test]

                    e = np.vstack((e_x, e_y))
                    total_error = e ** 2

                    rmse_test = np.sqrt(np.sum(total_error) / len(total_error))
                    print rmse_test

                    scores[i, k] = rmse_test
                    i += 1

# save the scores of each model in each test subset
np.savetxt("scores.csv", scores, delimiter=",", fmt='%.3f')
# save the list of all the experimented settings
with open("params.csv", 'w') as f:
    for param in hyperparameters:
        f.write("%d %f %d %s\n" % (param[0], param[1], param[2], param[3]))

# find best configuration
means = scores.mean(axis=1)
best_conf = hyperparameters[means.argmin()]

# retrain with the best configuration
ds = DenseDesignMatrix(X=X, y=y)

if best_conf[3] == "rectified":
    hidden_layer1 = mlp.RectifiedLinear(layer_name='hidden1', dim=best_conf[0], irange=.1, init_bias=1.)
else:
    if best_conf[3] == "tanh":
        hidden_layer1 = mlp.Tanh(layer_name='hidden1', dim=best_conf[0], irange=.1, init_bias=1.)
    else:
        if best_conf[3] == "sigmoid":
            hidden_layer1 = mlp.Sigmoid(layer_name='hidden1', dim=best_conf[0], irange=.1, init_bias=1.)

# statically cast second hidden layer to Tanh because it achieved a better performance
hidden_layer2 = mlp.Tanh(layer_name='hidden2', dim=dim2, irange=.1, init_bias=1.)
output_layer = mlp.Linear(dim=2, layer_name='y', irange=.1)

default = Default()
wdecay = WeightDecay(coeffs={'hidden1': 0.0001, 'hidden2': 0.0001, 'y': 0.0001})
costs = SumOfCosts([default, wdecay])

trainer = sgd.SGD(learning_rate=best_conf[1], batch_size=batch_size,
                  termination_criterion=EpochCounter(best_conf[2]), cost=costs)

layers = [hidden_layer1, hidden_layer2, output_layer]
ann = mlp.MLP(layers, nvis=inputs)
trainer.setup(ann, ds)

while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()

    if not trainer.continue_learning(ann):
        break

# beginning of the construction of the submission
X = feats_te
prediction = ann.fprop(theano.shared(X, name='inputs')).eval()
predict_x_te = prediction[:, 0]
predict_y_te = prediction[:, 1]
predict_x_te = predict_x_te.reshape(len(predict_x_te), 1)
predict_y_te = predict_y_te.reshape(len(predict_y_te), 1)

IDs = range(len(predict_x_te) + len(predict_y_te))

all_pred_te = np.vstack((predict_x_te, predict_y_te))

fw = open('nnCV.csv', 'w')
fw.write('Id,Prediction\n')
for i in IDs:
    fw.write('%d,%.5f\n' % (IDs[i] + 1, all_pred_te[i][0]))
fw.close()

print "mean rmse best configuration:", np.amax(means)
print "the best configuration is:", best_conf

