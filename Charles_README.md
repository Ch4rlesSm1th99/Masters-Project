# Project Aims and Data Notes

The goal of this project is to train a model to detect auroral events using data collected from the SuperDARN radar 
network in 1995. We may consider adding data from other years if more data is needed for the model to perform well.
The dataset is mostly unlabelled, except for a document from a PhD student noting 20 specific days when an auroral event
was detected. Since we have very few labels, two main approaches have been considered:

1. **Self-Supervised Approach**: Although this approach has not been applied to space plasma physics in the literature, it has 
shown promise in related time series tasks. We plan to try self-supervised learning first and see how it performs.
2. **Unsupervised Approach**: This is a common approach in the literature for similar problems, where dynamic clustering 
algorithms like DBSCAN or Gaussian Mixture Models (GMM) are used to detect or classify events.

   
The dataset contains around 2000 time series files with three key features that may indicate an auroral event: **power**, 
**velocity**, and **spectral width**.


# Self-Supervised Strategy


## 1. Modality Treatment
To give the model as much context as possible, we will treat the three features (power, velocity, spectral width) as 
separate modalities. This approach will give us more flexibility in how we combine these different types of data.

We will split the time series into smaller segments before feeding them into the model. The length of these segments 
needs to be longer than the typical duration of an auroral event. We may need to fine-tune the segment length to help 
the model learn effectively.

## 2. Data Augmentation
We will create augmented versions of the data by adding noise or shifting the data slightly. The goal is to generate 
alternate views of the same data while preserving global patterns. We will visualise the original data and its 
augmentations to ensure consistency and avoid training on bad data.

Data augmentation will be applied in the dataloader (ideally an argument of the dataloader), rather than creating a 
separate augmented dataset, to keep the process flexible for both training and testing.

## 3. Feature Extraction
Each modality will be passed through a 1D Convolutional Neural Network (1D CNN). We will train the CNN from scratch 
since the dataset is very specific to this problem. Pretrained models or fine-tuning may not work well here due to the 
uniqueness of the dataset.


## 4. Modality Fusion

After extracting features for each modality, we will concatenate them to form a combined feature space representing all 
three types of data. We may also experiment with hierarchical fusion to see which features contribute most to the 
detection.

## 5. Self-Supervised Strategy
We plan to use a contrastive loss function (e.g., SimCLR or Orchestra) or a triplet loss function to align similar pairs 
of augmented data and push apart dissimilar pairs whcih are created by randomly contrasting the positive pairs of data 
against the rest of the pairs in batch. This will effectively generate labels for the data and allow us to train the 
model in a way that cheeses supervised learning, despite having no explicit labels.

One challenge is that many self-supervised methods rely on batch-level accuracy, which might not give a true measure of 
performance on the unaugmented test data. We may also face memory limitations during testing due to the need for large 
batch sizes for models like SimCLR to learn.

At this stage, the model will be able to detect specific events, although it may not be able to classify them. For 
classification, we would need to generate a pseudo-labelled dataset.

## 6. Pseudo Labelling (Optional if persueing this model further)

If we decide to take this further, we could use the PhD student's document noting the 20 days with events as a starting 
point to create classes and compile a pseudo-labelled dataset. This could help train a simpler classification model.

# Unsupervised strategy .... (to be continued)
May explore in the future ....




