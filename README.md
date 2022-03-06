# s2t-sim

Source to Target Similarity Filtering for Unsupervised Domain Adaptation

## Install

`pip install -r requirements.txt`

## Data Preperation

To download DomainNet Data run:

`data/domainnet/download_data.sh`

To download Digit-Five Dataset, reffer to [1] or use the link the authors provided:

`https://drive.google.com/open?id=1A4RJOFj4BJkmliiEL7g9WzNIDUHLxfmm`

The datafiles used are stored in the \_img.txt files in 'data/domainnet/txt/' and utilized in the utils/dataset.py

### Reference

If you consider using this code or its derivatives, please consider citing:

```
@misc{
  title={Source to Target Similarity Filtering for Unsupervised Domain Adaptation},
  author={Noll, Jonas},
  year={2022}
}
```
