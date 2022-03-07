# s2t-sim: Source to Target Similarity Filtering for Unsupervised Domain Adaptation

## Install

Python 3.7:

`pip install -r requirements.txt`

## Data Preperation

To download DomainNet Data run:

`data/domainnet/download_data.sh`

The datafiles used for experimentation are stored in the \_img.txt files in 'data/domainnet/txt/' and utilized in the utils/dataset.py.

To download Digit-Five Dataset, reffer to [1] or use the link the authors provided:

`https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm`

The datafiles 'usps.h5', 'syn_number.mat' and 'mnistm_with_label.mat' need to be placed in 'data/digit-5/. Neglect MNIST & SVHN as the datasets are drawn from the pytorch library. If the data files are used from other sources or directories, adjust default data paths of the custom datasets in utils/dataset.py accordingly.

## Execution

The entry point to the code and experimentation pipeline is the **main.py**, run:

`python main.py`

Arguments are parsed in the opts.py file and can be adjusted either within the file or by passing arguments to the execution command.

## Results

Modeling results for the different sampling mehtods 'random', 'ssim', 'feature distance' and 'autoencoder loss' are stored under results/ and hold the experiment id (--exp_id) as a filename. Likewise, logs are saved under logs/.

### Reference

If you consider using this code or its derivatives, please consider citing:

```
@misc{
  title={Source to Target Similarity Filtering for Unsupervised Domain Adaptation},
  author={Noll, Jonas},
  year={2022}
}
```

[1] X. Peng, Q. Bai, X. Xia, Z. Huang, K. Saenko, and B. Wang. Moment matching formulti-source domain adaptation. Proceedings of the IEEE International Conferenceon Computer Vision, pages 1406–1415, 2019.
