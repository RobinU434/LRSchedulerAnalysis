# DeepLearningResearchKitchen

Contains project work and slides for the seminar topic of learning-rate schedulers

# LR Schedules [Ref : Frank, June 6th]

**Description:** Learning rate schedules are highly effective empirical tricks to improve
the training process of neural networks.

## Questions

- Why do we schedule learning rates in DL? What (categories) of LR schedules are used in practice?
- What other hyperparameters are scheduled and why?

## Papers

- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983v5)
- [A Closer Look at Deep Learning Heuristics: Learning rate restarts, Warmup and Distillation](https://arxiv.org/abs/1810.13243)
- [A disciplined approach to neural network hyper-parameters: Part 1](https://arxiv.org/abs/1803.09820v2)
- Section 2 of [Benchmarking Neural Network Training Algorithms](https://arxiv.org/abs/2306.07179)


## Sample experiment: 

For a new (more realistic) DL problem (e.g. a ViT/nanoGPT/diffusion model/AlgoPerf workload) experiment with different learning rate schedules. Try to design one that reaches a competitive target in the least amount of time.

## Stretch goal: 

Look into "infinite" learning rate schedules (ones where the training step budget is unknown).
