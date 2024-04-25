To hear some inpaints, use the scripts/inpaint_demo.ipynb notebook

To do method 1, download the dataset from https://huggingface.co/datasets/teticio/audio-diffusion-breaks-256, and run

```
python -m scripts.inpaint_attack
```

To train the neural network, run

```
python -m inpaint_nn.train
```

The evaluation method 1 can be found in scripts/inapint_evaluation.ipynb

The audiodiffusion directory contains the target model I use to run my experiments on
