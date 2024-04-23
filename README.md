## TSKD

Pytorch Code of  TSKD for Cross-modality  Person Re-Identification (ReID)

### Highlight

---

We propose a TSKDy for cross-modality person ReID. Our method used two-stage feature alignment, where the separated optimizations realize both inter/intra feature alignment.


#### Results on two standard benchmarks 

| Dataset       | mode            | Rank@1  | mAP     |
| ------------- | --------------- | ------- | ------- |
| SYSU-MM01 [1] | All-Search      | ~ 76.6% | ~ 73.0% |
| SYSU-MM01     | Indoor-Search   | ~ 82.7% | ~ 85.3% |
| RegDB [2]     | Visible-Thermal | ~ 91.1% | ~ 81.7% |
| RegDB         | Thermal-Visible | ~ 89.9% | ~ 80.5% |

*The code has been tested in Python 3.7, Pytorch = 1.1.0. Both of these two datasets may have some fluctuation due to random spliting.

#### 1. Datasets

- (1) SYSU-MM01 Dataset [1] : The SYSU-MM01 dataset is a large-scale and challenging RGB-IR cross-modality Re-ID dataset. It is collected from six camera views (four RGB and two near-infrared), including both indoor and outdoor environments. Overall, this dataset contains 287,628 RGB images and 15,792 IR images with 491 identities in total. 
  - run ``python pre_process_sysu.py``  after download the dataset, the trainning data will be stored in ".npy" format.
- (2) RegDB Dataset [2] : The RegDB dataset contains 412 identities, where each identity has 10 RGB images and 10 IR images. This dataset is randomly divided, with half for training and the remainings for testing.

#### 2. Training

Train a model by

```python
python run.py --dataset sysu --lr 0.1 batch-size 8 --sm_w 1 --md_w 0.05 --gpu 0  
```

- ``--dataset``: which dataset "sysu" or "regdb".
- ``--lr``: initial learning rate.
- ``--sm_w``: the weight of the self-mimic loss.
- ``--md_w``: the weight of the mutual distillation loss
- ``--gpu``: which gpu to run.

You may need manually define the data path first.

#### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by

```python
python test.py --dataset sysu --mode all --gpu 0 --resume 'model_path'
```

- ``--dataset``: which dataset "sysu" or "regdb".
- ``--mode``: "all" or "indoor" all search or indoor search (only for sysu dataset)
- ``--resume``: the saved model path. 
- ``--gpu``: which gpu to run.

#### 4. Reference

[1]  

[2] 
