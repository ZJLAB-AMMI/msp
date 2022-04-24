This project contains the source code and dataset to reproduce the main result of the paper 
"[On Tracking Dialogue State by Inheriting Slot Values in Mentioned Slot 
Pools](https://arxiv.org/abs/2202.07156)". 

This paper has been 
accepted as a poster paper of IJCAI-ECAI 2022.

We have tested the validity of this project at a machine equipped with Ubuntu 18.04.3 LTS system,
an Intel Xeon Silver 4214 CPU, 128 GB memory, and 8 Nvidia Tesla V100 video cards.

### Requirements
Please create a virtual python environment (or conda environment), and then install all packages 
listed in the "requirements.txt" into the environment.

The training phase of this model (in the default setting) requires about 14GB video memory. 
If your available video memory is less than 14GB, please decrease the batch size to an appropriate value 
(by using the command "python train.py --batch_size=*").

### Execute
1. Please uncompress the multiwoz21.zip file.
   
   The directory should be like:

```
   msp/
     train.py
     multiwoz21.zip
     ...
     muliwoz21/
       data.json
       ...
```

2. Activate the environment you just created.
3. Please run the train.py script to train and test the model.
The result will be saved in the /evaluation folder.

### Note
1. The data preprocessing procedure and pretrained model download procedure may take a long time
when you run the script first time, please wait patiently. 
2. You can set the "use_multi_gpu" to True (by using the command "python train.py --multi_gpu=True") 
to accelerate the training speed if you have multiple video cards.
3. We did not upload the MultiWOZ 2.2 dataset into the project to save space. Please follow the 
instructions below if you want to train and test the model via the MultiWOZ 2.2 dataset.
   1. Download the MultiWoZ 2.2 dataset from the [project](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2).
   2. Follow the instruction on the [project](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2)
   , converting the MultiWOZ 2.2 dataset to the MultiWOZ 2.1 dataset format.
   3. Save the restructured dataset into a folder named "multiwoz22" under the "msp" directory.
   4. Modify the value of the 9th line of config.py from "multiwoz21" to "multiwoz22".
   5. run the "train.py" script.

### Citation
Please cite our paper if you think our study is helpful to your research.
```
@article{sun2022tracking,
  title={On Tracking Dialogue State by Inheriting Slot Values in Mentioned Slot Pools},
  author={Sun, Zhoujian and Huang, Zhengxing and Ding, Nai},
  journal={arXiv preprint arXiv:2202.07156},
  year={2022}
}
```