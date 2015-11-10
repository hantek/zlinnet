import os
import sys
import numpy
from scipy import ndimage
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from dataset import CIFAR10
from layer import StackedLayer
from classifier import LogisticRegression
from model import ClassicalAutoencoder, ZerobiasAutoencoder
from preprocess import SubtractMeanAndNormalizeH, PCA
from train import GraddescentMinibatch, Dropout
from params import save_params, load_params, set_params, get_params

from minimize import minimize

#######################
# SET SUPER PARAMETER #
#######################

pca_retain = 800
hid_layer_sizes = [4000, 1000, 4000]
batchsize = 100
zae_threshold=1.

momentum = 0.9
pretrain_lr_zae = 1e-3
pretrain_lr_lin = 1e-4
weightdecay = 1.0
pretrain_epc = 800

logreg_lr = 0.5
logreg_epc = 1000

finetune_lr = 5e-3
finetune_epc = 1000

print " "
print "pca_retain =", pca_retain
print "hid_layer_sizes =", hid_layer_sizes
print "batchsize =", batchsize
print "zae_threshold =", zae_threshold
print "momentum =", momentum
print "pretrain, zae:       lr = %f, epc = %d" % (pretrain_lr_zae, pretrain_epc)
print "pretrain, lin:       lr = %f, epc = %d, wd = %.3f" % (pretrain_lr_lin, pretrain_epc, weightdecay)
print "logistic regression: lr = %f, epc = %d" % (logreg_lr, logreg_epc)
print "finetune:            lr = %f, epc = %d" % (finetune_lr, finetune_epc)

#############
# LOAD DATA #
#############

cifar10_data = CIFAR10()
train_x, train_y = cifar10_data.get_train_set()
test_x, test_y = cifar10_data.get_test_set()

print "\n... pre-processing"
preprocess_model = SubtractMeanAndNormalizeH(train_x.shape[1])
map_fun = theano.function([preprocess_model.varin], preprocess_model.output())

pca_obj = PCA()
pca_obj.fit(map_fun(train_x), retain=pca_retain, whiten=True)
preprocess_model = preprocess_model + pca_obj.forward_layer
preprocess_function = theano.function([preprocess_model.varin], preprocess_model.output())

pcamapping = theano.function([pca_obj.forward_layer.varin], pca_obj.forward_layer.output())
pcaback = theano.function([pca_obj.backward_layer.varin], pca_obj.backward_layer.output())

train_x = preprocess_function(train_x)
test_x = preprocess_function(test_x)

feature_num = train_x.shape[0] * train_x.shape[1]

train_x = theano.shared(value=train_x, name='train_x', borrow=True)
train_y = theano.shared(value=train_y, name='train_y', borrow=True)
test_x = theano.shared(value=test_x, name='test_x', borrow=True)
test_y = theano.shared(value=test_y, name='test_y', borrow=True)
print "Done."

#########################
# BUILD PRE-TRAIN MODEL #
#########################

print "... building pre-train model"
npy_rng = numpy.random.RandomState(123)
model = ZerobiasAutoencoder(
    train_x.get_value().shape[1], hid_layer_sizes[0], 
    init_w = theano.shared(
        value=0.01 * train_x.get_value()[:hid_layer_sizes[0], :].T,
        name='w_zae_0',
        borrow=True
    ),
    threshold=zae_threshold, vistype='real', tie=True, npy_rng=npy_rng
) + SubtractMeanAndNormalizeH(hid_layer_sizes[0]
) + ClassicalAutoencoder(
    hid_layer_sizes[0], hid_layer_sizes[1],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[0] * hid_layer_sizes[1] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[0] * hid_layer_sizes[1])].reshape(
            hid_layer_sizes[0], hid_layer_sizes[1]
        ),
        name='w_ae_1',
        borrow=True
    ),
    vistype = 'real', tie=True, npy_rng=npy_rng
) + SubtractMeanAndNormalizeH(hid_layer_sizes[1]
) + ZerobiasAutoencoder(
    hid_layer_sizes[1], hid_layer_sizes[2],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[1] * hid_layer_sizes[2] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[1] * hid_layer_sizes[2])].reshape(
            hid_layer_sizes[1], hid_layer_sizes[2]
        ),
        name='w_zae_2',
        borrow=True
    ),
    threshold=zae_threshold, vistype='real', tie=True, npy_rng=npy_rng
)
model.models_stack[2].params = [model.models_stack[2].w]
model.models_stack[2].params_private = [model.models_stack[2].w, model.models_stack[2].bT]

model.print_layer()
print "Done."

#############
# PRE-TRAIN #
#############

theano_rng = RandomStreams(123)
for i in range(0, len(model.models_stack), 2):
    if (i + 2) % 4 == 0:
        model.models_stack[i-2].threshold = 0.
        model.models_stack[i-1].varin = model.models_stack[i-2].output()

    print "\n\nPre-training layer %d:" % i
    layer_dropout = Dropout(model.models_stack[i], droprates=[0.2, 0.5], theano_rng=theano_rng).dropout_model
    layer_dropout.varin = model.models_stack[i].varin

    if (i + 2) % 4 == 0:
        model.models_stack[i-2].threshold = 0.
        pretrain_lr = pretrain_lr_lin
        layer_cost = layer_dropout.cost() + layer_dropout.contraction(weightdecay)
    else:
        pretrain_lr = pretrain_lr_zae
        layer_cost = layer_dropout.cost()

    trainer = GraddescentMinibatch(
        varin=model.varin, data=train_x,
        cost=layer_cost,
        params=layer_dropout.params_private,
        supervised=False,
        batchsize=batchsize, learningrate=pretrain_lr, momentum=momentum,
        rng=npy_rng
    )

    prev_cost = numpy.inf
    patience = 0
    origin_x = train_x.get_value()
    reshaped_x = pcaback(origin_x).reshape((50000, 32, 32, 3), order='F')
    for epoch in xrange(pretrain_epc):
        cost = 0.
        # original data
        cost += trainer.epoch()
        
        # data augmentation: horizontal flip
        flipped_x = reshaped_x[:, ::-1, :, :].reshape((50000, 3072), order='F')
        train_x.set_value(pcamapping(flipped_x))
        cost += trainer.epoch()

        # random rotation
        movies = numpy.zeros(reshaped_x.shape, dtype=theano.config.floatX)
        for i, im in enumerate(reshaped_x):
            angle_delta = (npy_rng.vonmises(0.0, 1.0)/(4 * numpy.pi)) * 180
            movies[i] = ndimage.rotate(im, angle_delta, reshape=False, mode='wrap')
        train_x.set_value(pcamapping(movies.reshape((50000, 3072), order='F')))
        cost += trainer.epoch()

        # random shift
        for i, im in enumerate(reshaped_x):
            shifts = (npy_rng.randint(-4, 4), npy_rng.randint(-4, 4), 0)
            movies[i] = ndimage.interpolation.shift(im, shifts, mode='reflect')
        train_x.set_value(pcamapping(movies.reshape((50000, 3072), order='F')))
        cost += trainer.epoch()

        cost /= 4.
        train_x.set_value(origin_x)
        
        if prev_cost <= cost:
            patience += 1
            if patience > 10:
                patience = 0
                trainer.set_learningrate(0.9 * trainer.learningrate)
            if trainer.learningrate < 1e-10:
                break
        prev_cost = cost
save_params(model, 'ZLIN_4000_1000_4000_normhid_nolinb_cae1_dtagmt2_dropout.npy')
print "Done."


#########################
# BUILD FINE-TUNE MODEL #
#########################

print "\n\n... building fine-tune model -- contraction 1"
for imodel in model.models_stack:
    imodel.threshold = 0.
model_ft = model + LogisticRegression(
    hid_layer_sizes[-1], 10, npy_rng=npy_rng
)
model_ft.print_layer()

train_set_error_rate = theano.function(
    [],
    T.mean(T.neq(model_ft.models_stack[-1].predict(), train_y)),
    givens = {model_ft.varin : train_x},
)
test_set_error_rate = theano.function(
    [],
    T.mean(T.neq(model_ft.models_stack[-1].predict(), test_y)),
    givens = {model_ft.varin : test_x},
)
print "Done."

print "... training with conjugate gradient: minimize.py"
fun_cost = theano.function(
    [model_ft.varin, model_ft.models_stack[-1].vartruth],
    model_ft.models_stack[-1].cost() + model_ft.models_stack[-1].weightdecay(weightdecay)
)
def return_cost(test_params, input_x, truth_y):
    tmp = get_params(model_ft.models_stack[-1])
    set_params(model_ft.models_stack[-1], test_params)
    result = fun_cost(input_x, truth_y)
    set_params(model_ft.models_stack[-1], tmp)
    return result

fun_grad = theano.function(
    [model_ft.varin, model_ft.models_stack[-1].vartruth],
    T.grad(model_ft.models_stack[-1].cost() + model_ft.models_stack[-1].weightdecay(weightdecay),
           model_ft.models_stack[-1].params)
)
def return_grad(test_params, input_x, truth_y):
    tmp = get_params(model_ft.models_stack[-1])
    set_params(model_ft.models_stack[-1], test_params)
    result = numpy.concatenate([numpy.array(i).flatten() for i in fun_grad(input_x, truth_y)])
    set_params(model_ft.models_stack[-1], tmp)
    return result
p, g, numlinesearches = minimize(
    get_params(model_ft.models_stack[-1]), return_cost, return_grad,
    (train_x.get_value(), train_y.get_value()), logreg_epc, verbose=False
)
set_params(model_ft.models_stack[-1], p)
save_params(model_ft, 'ZLIN_4000_1000_4000_10_normhid_nolinb_cae1_dtagmt2_dropout.npy')

load_params(model_ft, 'ZLIN_4000_1000_4000_10_normhid_nolinb_cae1_dtagmt2_dropout.npy')
print "***error rate: train: %f, test: %f" % (
    train_set_error_rate(), test_set_error_rate()
)

#############
# FINE-TUNE #
#############

"""
print "\n\n... fine-tuning the whole network"
truth = T.lmatrix('truth')
trainer = GraddescentMinibatch(
    varin=model_ft.varin, data=train_x, 
    truth=model_ft.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model_ft.models_stack[-1].cost(), 
    params=model.params,
    batchsize=batchsize, learningrate=finetune_lr, momentum=momentum,
    rng=npy_rng
)

prev_cost = numpy.inf
for epoch in xrange(finetune_epc):
    cost = trainer.epoch()
    if epoch % 100 == 0 and epoch != 0:  # prev_cost <= cost:
        trainer.set_learningrate(trainer.learningrate*0.8)
    if epoch % 50 == 0:
        print "***error rate: train: %f, test: %f" % (
            train_set_error_rate(), test_set_error_rate()
        )
    prev_cost = cost
print "Done."
"""



print "\n\n... fine-tuning the whole network, with dropout"
theano_rng = RandomStreams(123)
dropout_ft = Dropout(model_ft, droprates=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], theano_rng=theano_rng).dropout_model
dropout_ft.print_layer()

trainer = GraddescentMinibatch(
    varin=dropout_ft.varin, data=train_x, 
    truth=dropout_ft.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=dropout_ft.models_stack[-1].cost(),
    params=dropout_ft.params,
    batchsize=batchsize, learningrate=finetune_lr, momentum=momentum,
    rng=npy_rng
)

origin_x = train_x.get_value()
reshaped_x = pcaback(origin_x).reshape((50000, 32, 32, 3), order='F')
prev_cost = numpy.inf
patience = 0
for epoch in xrange(1000):
    cost = 0.
    # original data
    cost += trainer.epoch()
        
    # data augmentation: horizontal flip
    flipped_x = reshaped_x[:, ::-1, :, :].reshape((50000, 3072), order='F')
    train_x.set_value(pcamapping(flipped_x))
    cost += trainer.epoch()

    # random rotation
    movies = numpy.zeros(reshaped_x.shape, dtype=theano.config.floatX)
    for i, im in enumerate(reshaped_x):
        angle_delta = (npy_rng.vonmises(0.0, 1.0)/(4 * numpy.pi)) * 180
        movies[i] = ndimage.rotate(im, angle_delta, reshape=False, mode='wrap')
    train_x.set_value(pcamapping(movies.reshape((50000, 3072), order='F')))
    cost += trainer.epoch()

    # random shift
    for i, im in enumerate(reshaped_x):
        shifts = (npy_rng.randint(-4, 4), npy_rng.randint(-4, 4), 0)
        movies[i] = ndimage.interpolation.shift(im, shifts, mode='reflect')
    train_x.set_value(pcamapping(movies.reshape((50000, 3072), order='F')))
    cost += trainer.epoch()

    cost /= 4.
    train_x.set_value(origin_x)
        
    if prev_cost <= cost:
        patience += 1
        if patience > 5:
            patience = 0
            trainer.set_learningrate(0.9 * trainer.learningrate)
        if trainer.learningrate < 1e-10:
            break
    print "***error rate: train: %f, test: %f" % (train_set_error_rate(), test_set_error_rate())
    prev_cost = cost
print "Done."

print "***FINAL error rate, train: %f, test: %f" % (
    train_set_error_rate(), test_set_error_rate()
)
save_params(model_ft, 'ZLIN_4000_1000_4000_10_normhid_nolinb_cae1_dtagmt2_dropout_dpft.npy')
