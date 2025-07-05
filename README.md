### MeNet

MeNet is a mixed-effects deep neural network architecture for multi-environment agronomic traits prediction. This repository includes model deployment, training steps, prediction steps, and model parameter settings.

<img src="saved/MeNet.png" alt="MeNet Overview" style="zoom: 33%;" />

### Set up

Set up environments using the following command:

```python
conda create -n MeNet python=3.9.17
conda activate MeNet
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt 
```

### Data Preparation

- The genotype data and phenotype datasets should be in a data folder, organizing as follows:

  ```python
  + data
      + gene
      	++ genotype.csv
      + phen 
       	++ phenotype.csv
      + splits
  ```

- To generate training samples, validation samples and test samples, using the following command:

  ```python
  cd MeNet
  python utils/process_data/split.py
  ```



### RepGeno to generate genetic relatedness

- To generate genetic relatedness, train the backbone of the RepGeno. The resulting genetic relatedness is then saved in `saved/genetic_relatedness.csv`.

  ```python
  python generate_genetic_relatedness.py [parameters]
  
  [parameters]:
      --phenotype  # Phenotype name
      --windows_mechanism   # Windows mechanism 0: don't use windows_mechanism, 1: windows_mechanism by chromosome
      --windows_chr   # Number of chromosomes in a window
      --device	# Device: cuda or cpu
  ```

  Although we provide recommended [parameter settings](configs/RepGeno.json), considering differences in hardware, users are advised to carefully determine these parameters based on error variations during training. The constructed genetic relatedness will have a significant impact on the prediction performance of MeNet.



### Train MeNet and analyze the contribution between VE and RepGenp

- To train the MeNet model and evaluate the contributions of the VE and RepGeno modules, use the following command. The error tracking results during training and the contributions of the VE and RepGeno modules will be saved in `menet.log`

  ```python
  python train_menet.py [parameters] > menet.log 
  
  [parameters]:
      --phenotype  # Phenotype name
      --windows_mechanism   # Windows mechanism 0: don't use windows_mechanism, 1: windows_mechanism by chromosome
      --windows_chr   # Number of chromosomes in a window
      --device	# Device: cuda or cpu
  ```

​	Although we provide recommended [parameter settings](configs/MeNet.json), considering differences in hardware, users are advised to carefully determine these parameters based on loss during training. 



### Transfer Learning

- To apply transfer learning, you can fine-tune a pre-trained model by selectively freezing and unfreezing specific layers. The following code snippet demonstrates how to configure the model after loading a pre-trained model:

  ```python
  # Freeze all parameters in the model
  for param in model.parameters():
      param.requires_grad = False
  
  # Unfreeze 
  model.variation_lower.embedding.requires_grad_(True)
  model.variation_upper.ve[0].requires_grad_(True)
  
  # Unfreeze 
  for param in model.variation_output.parameters():
      param.requires_grad = True
  for param in model.gr_output.parameters():
      param.requires_grad = True
  for param in model.fusion.parameters():
      param.requires_grad = True
  ```

  ##### Explanation

  - **Freezing the model**: Setting `requires_grad = False` for all parameters prevents them from being updated during training, preserving the pre-trained weights.
  - **Selective unfreezing**: Specific layers are set to `requires_grad = True` to allow fine-tuning.
  - Ensure the pre-trained model is loaded before applying these settings.

### Acknowledgements

This work is based on [pytorch](https://pytorch.org/) ,  [scikit-learn](https://scikit-learn.org/),  and [plink](https://www.cog-genomics.org/plink/). The project is developed by following author and supervised by Prof. Xiangchao Gan(gan@njau.edu.cn)

Authors:

Yanhui Li  (huiyl@stu.njau.edu.cn): prototype development, data processing, model validation 

Shengjie Ren (sunflower@stu.njau.edu.cn): overall frame work, model construction, model optimization,  contribution analysis

