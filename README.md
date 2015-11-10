# zlinnet
This repo provides an implementation of Z-Lin network. It should reproduce the results on permutation invariant CIFAR-10 reported in the paper:
 - Zhouhan Lin, Roland Memisevic, Kishore Konda, [How far can we go without convolution: Improving fully-connected networks](http://arxiv.org/pdf/1511.02580v1.pdf), arXiv preprint arXiv:1511.02580 (2015).

## Setup
 - Download dependency. First you need to download NeuroBricks. It is a super light framework we use in this repo, which is currently not coverring Recurrent Nets and not documented. There are far more mature and successful ones available, like [blocks](https://github.com/mila-udem/blocks) and [lasagne](https://github.com/Lasagne/Lasagne). Execute the following line to download the source:

       git clone https://github.com/hantek/NeuroBricks.git

 - (Optional) In case anything changes in the future, checking out to the snapshot at the time when this repo is published should ensure everything working fine. It may still work without checking out to that snapshot, but just in case.

       git checkout 191704feb5de67ab2815d5891dd633b9f2d04afb

 - Import the path you store NeuroBlocks to python's searching directory.

 - Specify the data path to wherever you store your CIFAR10 dataset. At line 56 on both .py scripts, modify the line to:

       cifar10_data = CIFAR10(folderpath="/path/to/your/cifar-10-batches-py/folder")

Now you should be able to run the codes with no problem.

## Permutation invariant CIFAR-10

Execute the following commands in your terminal:

       python expr_cifar10_ZLIN_normhid_nolinb_dropout.py

It should generate an accuracy around 69.62% in the end.

## CIFAR-10 with deformations

It is basically the same model but with data augmentation. So it's a feed-forward, fully connected network. Type

       python expr_cifar10_ZLIN_normhid_nolinb_dtagmt_dropout.py

to execute. You can expect an 78.62% accuracy after training process finishes.


