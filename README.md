# Interferometric Graph Transform

This repository contains the code corresponding to the ICML-2020 publication: https://arxiv.org/abs/2006.05722

If you have some troubles to reproduce some results, some questions regarding the code or even find a bug, please open directly an issue 
in this repository. If your questions concern some technical elements of the paper, please send directly an email to the author, firstname.name[AT]lip6.fr

If you wish to cite this paper, please use:

```@inproceedings{oyallon2020interferometric,
  title={Interferometric Graph Transform: a Deep Unsupervised Graph Representation},
  author={Oyallon, Edouard},
  booktitle={37th International Conference on Machine Learning (ICML 2020)},
  year={2020}
}
```

## Reproducing the numerical results

In order to reproduce the experiments of the paper <i>Interferometric Graph Transform: a Deep Unsupervised Graph Representation</i>, 
make sure you have `pytorch` and `python 3` installed, as well as `MATLAB` (only for the Haar Scattering experiments) and please
 follow the next instructions.

### Community detection experiments

Please replace (or add) the files of https://github.com/alelab-upenn/graph-scattering-transforms with the ones in the folder "community-detection". Then run the code as described in this GitHub repository, by adding the corresponding dataset, as described.

### Images experiments

Please run the following instructions on a GPU:

#### CIFAR10/MNIST, graph

`$DATASET=cifar/mnist`, `$PATH_TO_MODEL` refers to the path in which the first script stores the weight of the model. 

```
python build_representation.py --K 40 --J 3 --lr_schedule '{0:1e0,500:1e-1,1000:1e-2,1500:1e-3}' --epochs 5  --dataset $DATASET
python classif.py --J 3 --feature interferometric --C 1e-3 --mode 1 --classifier 'svm'  --dataset $DATASET --path $PATH_TO_MODEL
```

#### CIFAR10, scattering

```
python classif.py --J 3 --feature scattering --C 1e-3 --mode 1 --classifier 'svm'  --dataset cifar
```

### Graph experiments

#### NTU/SBU

Move the dataset (available <a href="#">here</a>) in the same folder as the main file, then `$DATASET=NTU_xview_values/NTU_xsub_values/SBU`,
`$PATH_TO_MODEL` refers to the path in which the first script stores the weight of the model, `$ORDER` is the required order and 
`$K` the respective number of filters, which should end by 0 (e.g., `$ORDER=1` leads to `$K=30,0` for instance).

Finally, `$LR_SCHEDULE` must be the learning rate schedule reported in the paper (as above). Add `--no_averaging` to remove the averaging from 
the experiments, yet this will substantially increase the computation time because the representation is larger.

```
python build_representation_graph.py --K $K --order $ORDER --lr_schedule $LR_SCHEDULE --epochs 5 --data_name $DATASET
python classif_graph.py --mode 'svm' --order 1 --data_name $DATASET --file $PATH_TO_MODEL  --K 10,5,0 --C 1e-1 
```

#### Haar scattering

First download ``HaarScat`` from  https://www.di.ens.fr/data/software/ and then switch and run the scripts in the folder demo that we provided.
