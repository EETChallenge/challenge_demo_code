
# [Event-based Eye Tracking Challenge](https://EETChallenge.github.io/EET.github.io/) 

**[Zuowen Wang<sup>1</sup>](https://scholar.google.com/citations?user=pdZLukIAAAAJ&hl=en), [Chang Gao<sup>2</sup>](https://scholar.google.com/citations?user=sQ9N7dsAAAAJ&hl=en), [Zongwei Wu<sup>3</sup>](https://scholar.google.fr/citations?user=3QSALjX498QC&hl=en&oi=en), [Shih-Chii Liu<sup>1</sup>](https://scholar.google.com/citations?user=XYkPvZUAAAAJ&hl=en), [Qinyu Chen<sup>1,4</sup>](https://scholar.google.com/citations?user=enuSO2YAAAAJ&hl=en)**


[1. Sensors Group, Institute of Neuroinformatics, UZH/ETH Zurich](https://sensors.ini.ch/)

[2. EMI Lab, TU Delft](https://www.tudemi.com/)

[3. Computer Vision Lab, University of Wurzburg](https://www.informatik.uni-wuerzburg.de/computervision/)

[4. Leiden Institute of Advanced Computer Science, Leiden University](https://scholar.google.com/citations?user=enuSO2YAAAAJ&hl=en)


## @ CVPR 2024 Workshop [AI for Streaming](https://ai4streaming-workshop.github.io/)


<!-- 
🚀 🚀 🚀 **News**

-  ["Efficient Deep Models for Real-Time 4K Image Super-Resolution. NTIRE 2023 Benchmark and Report"](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Conde_Efficient_Deep_Models_for_Real-Time_4K_Image_Super-Resolution._NTIRE_2023_CVPRW_2023_paper.pdf)
- ["Towards Real-Time 4K Image Super-Resolution"](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zamfir_Towards_Real-Time_4K_Image_Super-Resolution_CVPRW_2023_paper.pdf)
- [Project website](https://eduardzamfir.github.io/NTIRE23-RTSR/) 
- Dataset release soon (mid-June) -->
<!-- - Presentation June 18th NTIRE workshop. -->

<details>
<summary><b>Citation</b></summary>
<p><code>

@INPROCEEDINGS{3et,
  author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
  booktitle={2023 IEEE Biomedical Circuits and Systems Conference (BioCAS)}, 
  title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network}, 
  year={2023},
  pages={1-5},
  doi={10.1109/BioCAS58349.2023.10389062}}
  
</code></p>

</details> 

<br>

----


## About the Challenge
Developing an event-based eye-tracking system presents significant opportunities in diverse fields, notably in consumer electronics and neuroscience. Human eyes exhibit rapid movements, occasionally surpassing speeds of 300°/s. This necessitates using [event cameras](https://www.youtube.com/watch?v=6xOmo7Ikwzk&t=80s&ab_channel=Sony-Global) capable of high-speed sampling and tracking. 

<figure align="center">
    <img src="./figures/3et_demo_low_res.gif" width=480>
    <figcaption>Figure 1. Let's play some video game with event-based eye-tracking!</figcaption>
</figure>


In consumer electronics, particularly in augmented and virtual reality (AR/VR) applications, the primary benefits of event-based systems extend beyond their high speed. Their highly sparse input data streams can be exploited to reduce power consumption. This is a pivotal advantage in creating lighter, more efficient wearable headsets that offer prolonged usage and enhanced user comfort. 

This is instrumental in augmenting the immersive experience in AR/VR and expanding the capabilities of portable technology. In neuroscience and cognitive studies, such technology is crucial for deciphering the complexities of eye movement. It facilitates a deeper comprehension of visual attention processes and aids in diagnosing and understanding neurological disorders. 

This challenge aims to develop an **event-based eye-tracking system for precise tracking of rapid eye movements** to produce lighter and more comfortable devices for a better user experience. Simultaneously, it promises to provide novel insights into neuroscience and cognitive research, deepening our understanding of these domains.

----
## Start Training!
We provide a handy training script for you to start with. Simply install the dependencies in the environment.yml file with conda and run the following command:
```python
python3 train.py --config sliced_baseline.json
```
It should give a decent baseline performance. Play around with the hyperparameters and see if you can improve the performance! 

For generating the results for submission we also provide a test script test.py. Please refer to the section [Prepare test results and submission](#Prepare-test-results-and-submission) for more details.


----
## **Dataset Description**
There are 13 subjects in total, each having 2-6 recording sessions. The subjects are required to perform 5 classes of activities: random, saccades, read text, smooth pursuit and blinks. Figure 2 visualizes one real recording sample by making the raw events into event frames. The total data volume is approximately 1 GB in the compressed .h5 form. 


<figure align="center">
    <img src="./figures/movie.gif" width="480" height="360">
    <figcaption>Figure 2. Eye movement filmed with event camera.</figcaption>
</figure>


#### **Dataloader**
We provide a very convienient dataloader for loading the dataset. The dataloader and transformations are based on the [Tonic](https://tonic.readthedocs.io/en/latest/index.html) event camera library. 

Preprocessing steps for event data is of particular importance and difficulty, since it is dependent on many aspects such as event representation, model input format requirement and the task itself. The dataloader we provide is a good starting point for the challenge. It slices the raw event recordings into strided sub-recordings and convert them into event voxel grids. It also enables local caching for the preprocessed data on the disk so that the program does not have to process the data everytime (But if it has different preprocessing parameter such as different stride, then it needs to be recached).

The event recordings are provided in the form of .h5 files of raw events. Each event is represented by a tuple of (t, x, y, p), where t represents the timestamp the event happened, (x, y) represents the spatial coordinate, and p represents the polarity of the event, +1 indicates the light intensity goes up and -1 indicates goes down. 

These raw events are loaded with
```python
train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform)
```
'transform' and 'target_transform' essentially do the following:

* downsample spatially by the factor of 8 on width and height, to lower the training hardware requirement for the challenge (originally 640x480 to 80x60).
* downsample the ground truth label frequency to 20Hz, this will be discussed in the next subsection ['Labeling'](#Labeling).

The challenger is **free to decide** whether to use the raw events in combination with models such as spiking neural networks or other methods, or to convert the raw events into event frames/ voxel grids and use them as input to the model, similiar to feeding an image to the model. 

In the following code sniplet we provide a common way of processing raw events and convert it into event voxel grids. 
```python
slicing_time_window = args.train_length*int(10000/temp_subsample_factor) #microseconds
train_stride_time = int(10000/temp_subsample_factor*args.train_stride) #microseconds

train_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window-train_stride_time, \
                seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False)
```
First we determine how to divide the raw recordings into sub-recordings. The 'slicing_time_window' is the length of each sub-recording, and the 'train_stride_time' is the stride between two consecutive sub-recordings. For example, if args.train_length=30 and temp_subsample_factor=0.2, then the slicing_time_window=30*(10000us/0.2)=1.5s. Meaning that each sub-recording is 1.5s long, and in this sequence, every event frame/ voxel grid will correspond to a recording time window of 10000us/0.2=50ms, and there will be 30 of them in this sub-sequence. Assume the args.train_stride is 5, then train_stride_time=5*(10000us/0.2)=250us, meaning that the next sub-recording will start 250us after the previous one. This is for expanding the total number of training samples.


After the raw event sequence is sliced into raw event sub-sequences, we can convert each of them into different event representations. The transformations are defined in the following code sniplet. **SliceLongEventsToShort** is a transformation that separate the raw event sub-sequences further into (10000us)/temp_subsample_factor time windows. **EventSlicesToVoxelGrid** is a transformation that convert each time window into the actual event representation, in this case, voxel grids with args.n_time_bins number of time bins. 


```python

post_slicer_transform = transforms.Compose([
    SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
    EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
                            n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
])

```

We then pass these transformations to the Tonic SlicedDataset class to post process the loaded raw events:
```python
train_data = SlicedDataset(train_data_orig, train_slicer, \
  transform=post_slicer_transform, metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
```

The SlicedDataset has a convenient function to cache the indices of how the raw events are sliced, when argument metadata_path is provided not None. But be careful if you provided the same metadata_path for different slicing strategies, the SlicedDataset will ignore the slicing parameters and use the old indices, causing unexpected results.


We can further cache the transformed voxel grid representation on the disk to further speed up data preprocessing. This is done by the DiskCachedDataset class. It will slow done the first time loading the data but for future epoch and future training, it will be much faster.
```python
train_data = DiskCachedDataset(train_data, \
  cache_path=f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}')

# at this point we can pass the dataset to the standard pytorch dataloader.
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=int(os.cpu_count()-2), pin_memory=True)
```

You can easily find a lot of [data augmentation](https://tonic.readthedocs.io/en/latest/auto_examples/index.html) methods in the [Tonic](https://tonic.readthedocs.io/en/latest/index.html) library and include them in the dataloader to further improve the performance of your model.


#### Labeling
The ground truth is labeled at 100Hz and consists of two parts for each label (x, y, close) with 

* labeling of the pupil center coordinates (x,y).

* a binary value 'close' indicating whether the eye blinks or not (0 for opening, 1 for closing). 

Labels (x,y,close) for the train split are provided at frequency of 100Hz. The user is free to decide at if they would like to downsample this frequency and whether to use the close label or not. In the training sample pipeline we provided, the labels are downsample to 20Hz and we do not use the close label by default. For the test set, we will evaluate the prediction at **20Hz** only. If you would like to alter the data loading pipeline, please be aware this downsampling.

#### Dataset splitting
We use 12 recordings for testing (test split) and the remaining recordings (train split) are for the user to train and validate their methods. The users are free to divide the training and validation sets from the training data. 

----


## **Performance of baseline methods**

#### Baseline performance with standard event voxel grids
We conducted benchmarking of the dataset with several baseline methods. The baseline methods are trained with the standard event voxel grid representation. The results are shown in the following table. The inference latency is measured on a single RTX 3090 GPU with batch size = 1.

| Method                                                                                    | GPU            | Averaged Distance  | P10 Accuracy       |  Inference Latency (bs=1)    | 
|-------------------------------------------------------------------------------------------|----------------|----------|------------------|----------|
|CNN_GRU | RTX 3090 24 Gb | 8.99    |0.63  | -    |



#### Monitoring and logging of training
We provide code sample together with [mlflow](https://mlflow.org/) for monitoring training progress and tasks. It will log the essnetial informations under the folder 'mlruns'. To monitor the runs, simply run the following command in the terminal:
```bash
mlflow ui --port 5000
```
Then open the browser and go to http://localhost:5000. You will see the following interface: 

![mlflow](./figures/mlflow.png)

<br>

----

##  **Prepare test results and submission**

We provide a sample script (test.py) to prepare your test results for submission. The script will generate a submission.csv file for submitting to the Kaggle challenge server.

**Please be aware** that if your data loading strategy is different from the standard strategy we provided, you need to modify the test set loader according to your method.


````
python .test.py --config [CONFIG_PATH] --checkpoint [CHECKPOINT_PATH] --output_path [OUTPUT_PATH]
````

If you trained your model with mlflow, the checkpoint should be saved in mlruns folder. You can find the checkpoint path in the mlflow ui under arfitacts section.

<br>

----

## **Evaluation of your submission**

We request that you submit a ```submission.csv``` file, which should contain two columns 'x' and 'y'. Notice that the range of x is [0, 80] and y is [0, 60].

#### scoring functions
There are two evaluation metrics for the challenge:
**p10 accuracy** (used for leaderboard on Kaggle) and **averaged euclidean distances** (also taken into consideration for the report).

The p10 accuracy is defined as the percentage of the predictions that are within 10 pixels of the ground truths (in the downsampled 80x60 spatial space). The averaged euclidean distance is defined as the average euclidean distance between the predictions and the ground truths. 

#### inference speed will be taken into consideration
We do not measure the latency of your method in the score board. But in the final report, the **inference speed (latency)** is required and is a **very important** metric for evaluating the final report. That is said if your method requires huge amount of computation, it is probably not a good method for practical real-time eye tracking applications.



----

## Citation and  acknowledgement

```
@inproceedings{chen20233et,
  title={3et: Efficient Event-based Eye Tracking Using a Change-based Convlstm Network},
  author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
  booktitle={2023 IEEE Biomedical Circuits and Systems Conference (BioCAS)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Contact

- Zuowen Wang (zuowen@ini.uzh.ch)
- Qinyu Chen (q.chen@liacs.leidenuniv.nl)
