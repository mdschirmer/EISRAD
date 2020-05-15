# EISRAD
EISRAD ({e}valuation of {i}mage {s}egmentations using {rad}ar plots)) is a tool to compare segmentations between raters based on multiple similarity metric in form of a radar plot. 

(EISRAD - Ice circle; natural phenomenon appearing on the Vigala River in Estonia)

![polar plot example](polar.png)

## Example usage
```
./eisrad.py -f segmentations.csv -o radar.png
```

## Description
Compares two segmentations of rater A (here called manual) and rater B (here called automated). The reported metrics are Dice coefficient (Dice), Jaccard index (Jaccard), true positive rate (TPR), volumetric similarity (VS), Mutual information (MI), Adjusted Rand Index (ARI), intraclass correlation coefficient (ICC), probabilistic distance (PBD), Cohens kappa (KAP), Detection Error Rate (DER) and Outline Error Rate (OER). The solid line is based on the median of each measure, while the ribbon represents the interquartile range.

Requires a csv file as input with two columns of the format: {manual_segmentation_file_path},{automated_segmentation_file_path}. Files should be in NIfTI/nii[.gz] format. The input segmentations can additionally binarized (>0) as part of the code. 

Output "-o" will be a png file radar plot '{your_output_file_name}.png' as demonstrated above. Additionally, using the "-r" flag, the metrics can be returned as a csv file. 

There are further formatting options for the colorbar (see below), including log-transforming the volumes of rater A. 

## Call options

Use './eisrad.py -h' or './eisrad.py --help' for descriptions of the optional parameters as below

```
Usage: eisrad.py [options]

Options:

  -h, --help                  show this help message and exit
  -f FILE, --file=FILE        Input FILE
  -o FILE, --output=FILE      Output image FILE.png
  -r FILE, --results=FILE     Output csv file with all measures
  -m MIN, --min=MIN           Minimum colorbar value
  -M MAX, --max=MAX           Maximum colorbar value
  -L STRING, --label=STRING   Label for colorbar
  -l, --log                   Plot logarithmic colorbar values
  -u STRING, --unit=STRING    Label for colorbar
  -d, --display               Display the output before saving as png
  -v, --verbose               verbose output
  -b, --binarize              binarize input images
```