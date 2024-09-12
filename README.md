![argmax.ai](argmaxlogo.png)

*This repository is published and maintained by the Volkswagen Group Machine Learning Research Lab.*

*Learn more at https://argmax.org.*


# Reference implementation of Deep Variational Bayes Filters

This repository contains the source code of our algorithm Deep Variational Bayes Filter that our publications base on.

# Usage instructions


## Initiate training

To start training following command needs to be called:

```bash
docker compose run train
```

Alternatively you can start training with gpu access by running:

```bash
docker compose run train-gpu
```

If necessary, this command will build the docker container and then start the training script.
To access training curves and visualisations you can start the tensorboard in a seperate terminal with the following command:

```bash
docker compose run tensorboard
```

# Related Publications

If you find the code or models in this repository useful for your research, please consider citing our work.
The method for learning sequence models described in this repository is used in the following [paper](https://arxiv.org/abs/2003.08876) and [dissertation](http://mediatum.ub.tum.de/1484075).

```BibTeX
@misc{beckerehmck2020learning,
  title={Learning to Fly via Deep Model-Based Reinforcement Learning}, 
  author={Philip Becker-Ehmck and Maximilian Karl and Jan Peters and Patrick van der Smagt},
  year={2020},
  eprint={2003.08876},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

```BibTeX
@phdthesis {karl2020unsupervisedcontrol,
  author={Maximilian Karl},
  title={Unsupervised Control},
  type={Dissertation},
  school={Technische Universität München},
  address={München},
  year={2020}
}
```


## Disclaimer

The purpose of this source code is limited to bare demonstration of the experimental section of the related papers.
