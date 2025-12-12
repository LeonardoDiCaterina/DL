# Deep Learning for Animal Species Recognition

**Author:** Leonardo Di Caterina

**Fine-grained image classification across 202 animal families using Convolutional Neural Networks**

---

## Overview

This project tackles the challenging problem of fine-grained image classification in the biological domain, specifically aiming to classify 9,586 images across 202 distinct animal families. Unlike curated academic datasets, this work addresses real-world challenges that emerge when working with data scraped from online encyclopedias, where images are noisy, uncurated, and often contain significant quality variations.

The primary goal is to develop a robust Convolutional Neural Network architecture capable of distinguishing between visually similar animal families despite the inherent difficulties present in the dataset. This research demonstrates how transfer learning and careful hyperparameter tuning can partially mitigate the challenges posed by imperfect data, providing insights valuable for practitioners working with similarly challenging real-world image classification tasks.

## The Challenge

Building an effective classifier for this dataset requires overcoming several significant obstacles that distinguish this work from typical benchmark experiments:

**Data Quality Issues:** The images were obtained through web scraping from an online encyclopedia, resulting in a dataset that includes not only photographs of living animals but also scientific diagrams, skeletal reconstructions, taxonomic illustrations, and historical drawings. These irrelevant images introduce substantial label noise, as they may be tagged with the correct family name but provide little useful visual information about the actual appearance of living members of that family. This heterogeneity in image content creates a fundamentally more difficult learning problem than would be encountered with carefully curated datasets.

**Extreme Class Imbalance:** The distribution of images across the 202 families follows a severe long-tail pattern, where some families are represented by hundreds of images while others have only a handful of examples. This imbalance means that standard training procedures would naturally bias the model toward predicting the most common classes, potentially leaving rare families essentially unlearnable. Addressing this imbalance while maintaining overall performance represents a core challenge in the experimental design.

**Variable Image Quality:** The scraped images exhibit substantial variation in resolution, aspect ratio, lighting conditions, and image quality. Some images are high-resolution professional photographs suitable for detailed feature extraction, while others are low-resolution thumbnails or degraded historical images. This variability in input quality requires the model to learn robust features that generalize across different imaging conditions rather than relying on spurious correlations with image quality.

**Architectural Constraints:** The project was deliberately constrained to use Convolutional Neural Networks rather than more recent Vision Transformer architectures. This constraint requires extracting maximum performance from CNN-based approaches through careful architecture selection, transfer learning, and hyperparameter optimization, providing insights into the practical limits of CNN-based methods for fine-grained classification tasks.

## Repository Structure

The repository is organized into logical modules that separate data processing, model training, evaluation utilities, and exploratory analysis. This structure facilitates reproducibility and allows each component of the pipeline to be understood and modified independently.

### Data Preprocessing Module

The `data_preprocessing/` directory contains all functionality related to preparing the raw image data for model training:

**`data_loading.py`** handles the ingestion of images from the scraped dataset, implementing efficient data loading pipelines that can handle the variable image sizes and formats present in the raw data. This module includes functionality for reading image metadata, handling corrupt or unreadable image files, and organizing the data according to family labels.

**`augmentation.py`** implements the data augmentation strategy designed to artificially increase dataset diversity and improve model robustness. The augmentation pipeline includes geometric transformations such as rotation, flipping, and cropping, as well as photometric adjustments like brightness and contrast variation. These augmentations help the model learn features that are invariant to common variations in imaging conditions while partially addressing the class imbalance problem by generating synthetic examples of underrepresented families.

**`splitting.py`** manages the partitioning of the dataset into training, validation, and test sets. This module implements stratified splitting to ensure that the severe class imbalance is proportionally represented in each split, preventing situations where rare families might be entirely absent from the validation or test sets.

### Training Module

The `training/` directory contains the core functionality for model selection, hyperparameter optimization, and model training:

**`model_selection_utils.py`** implements the comprehensive Grid Search procedure described in Section 4 of the project report. This systematic exploration evaluates combinations of learning rates, dense layer configurations, and dropout rates to identify optimal hyperparameters for each candidate architecture. The Grid Search is performed independently for each base model (ResNet50, MobileNetV2, VGG19) to account for architecture-specific optimization landscapes.

**`model_training_utils.py`** provides utilities for the actual training process, including functions for configuring optimizers, implementing learning rate schedules, managing callbacks for early stopping and model checkpointing, and logging training metrics. This module abstracts away the boilerplate code required for training loops, allowing the experimental notebooks to focus on high-level experimental design decisions.

### Custom Keras Components

The `Keras_custom/` directory contains custom implementations of loss functions and evaluation metrics tailored to the specific challenges of this classification problem. These custom components allow for more nuanced optimization objectives than standard categorical cross-entropy, potentially incorporating class weighting schemes or focal loss variants that emphasize learning on difficult or underrepresented examples.

### Notebooks

The `notebooks/` directory provides interactive Jupyter notebooks that document the complete experimental workflow from initial data exploration through final model training:

**`metadata_exploration.ipynb`** corresponds to Section 2 of the project report and provides a comprehensive analysis of the dataset characteristics. This notebook visualizes the distribution of images across families, examines the severity of class imbalance through histograms and cumulative distribution plots, investigates image resolution distributions, and identifies potential data quality issues. The insights gained from this exploratory analysis inform subsequent decisions about data augmentation strategies and model architecture choices.

**`colab_grid_search_routine copy.ipynb`** implements the systematic Grid Search workflow described in Section 4 of the report. This notebook trains multiple configurations of each candidate architecture on a reduced training set, evaluates their performance on a validation set, and records the results in a structured format for comparison. The Grid Search identifies promising hyperparameter configurations that balance training performance against validation performance, helping to select settings that generalize well rather than simply memorizing the training data.

**`colab_trainig_routine.ipynb`** contains the final model training procedure detailed in Section 4.1 of the report. This notebook takes the best hyperparameter configuration identified through Grid Search and trains the final model on the complete training set. It includes detailed monitoring of training and validation metrics across epochs, generation of learning curves to assess convergence behavior, and creation of the confusion matrix used to analyze per-class performance patterns.

**`image_modifications.ipynb`** provides visualization of the preprocessing and augmentation pipeline, showing side-by-side comparisons of original images and their augmented variants. This visualization helps verify that the augmentation strategy produces realistic variations without introducing artifacts that might harm model performance.

## Methodology

The experimental approach combines transfer learning with systematic hyperparameter optimization to extract maximum performance from CNN architectures despite the challenging dataset characteristics.

### Model Architectures Evaluated

Three established CNN architectures were selected as candidate base models, each representing different points in the accuracy-efficiency tradeoff space:

**ResNet50** employs residual connections that allow training of very deep networks by mitigating the vanishing gradient problem. The architecture's depth enables learning of hierarchical features at multiple scales, potentially beneficial for capturing the subtle differences between visually similar animal families. However, this depth also increases the risk of overfitting, particularly on datasets with limited examples per class.

**MobileNetV2** uses depthwise separable convolutions to achieve computational efficiency with fewer parameters than traditional CNNs. This efficiency makes MobileNetV2 attractive for deployment scenarios but may come at the cost of reduced representational capacity, potentially limiting performance on fine-grained classification tasks requiring discrimination of subtle visual differences.

**VGG19** represents a simpler architectural philosophy, relying on sequential convolutional layers with small receptive fields rather than more sophisticated architectural innovations. Despite being an older architecture, VGG19's straightforward design can provide a useful baseline for comparison and may exhibit different overfitting characteristics than more complex architectures.

### Transfer Learning Strategy

All three architectures were initialized with weights pretrained on ImageNet, a large-scale image classification dataset containing over one million images across one thousand categories. This transfer learning approach leverages the general visual features learned from ImageNet (such as edge detectors, texture patterns, and part detectors) as a starting point for learning animal family-specific features.

The pretrained convolutional layers were frozen during initial training phases, treating them as fixed feature extractors while only training newly added dense classification layers. This freezing strategy reduces the number of trainable parameters, helping to prevent overfitting when working with limited data while allowing the model to adapt the high-level feature representations to the specific characteristics of animal family classification.

### Hyperparameter Optimization

A model-wise Grid Search explored combinations of three key hyperparameters that control the capacity and regularization of the classifier:

The **learning rate** controls the step size of gradient descent updates during training. Too large a learning rate can cause training to diverge or oscillate without converging, while too small a learning rate results in excessively slow convergence or getting stuck in poor local minima. The Grid Search evaluated multiple learning rates to find values that enable efficient convergence to good solutions.

The **dense layer configuration** determines the capacity of the classifier head added on top of the frozen feature extractor. Wider dense layers can learn more complex decision boundaries but increase the risk of overfitting, particularly when the number of training examples is limited. The Grid Search explored different numbers and sizes of dense layers to balance expressiveness against generalization.

The **dropout rate** provides stochastic regularization by randomly setting a fraction of activations to zero during training, forcing the network to learn redundant representations that are less sensitive to the specific activations of individual neurons. Higher dropout rates provide stronger regularization but can slow convergence or prevent the model from learning sufficiently complex patterns. The Grid Search identified dropout rates that effectively regularize without excessively constraining the model.

### Final Model Refinement

After the Grid Search identified promising configurations, the final model training incorporated two additional refinements that improved convergence behavior. The batch size was increased from the Grid Search value to fifty examples per batch, providing more stable gradient estimates that reduce training noise and allow the model to converge more smoothly. Additionally, the learning rate was set to one times ten to the negative fourth power (1e-4), a value identified as effective during the Grid Search phase, balancing convergence speed against stability.

## Key Results

The experimental evaluation revealed distinct performance characteristics for each architecture and provided insights into the tradeoffs between training accuracy and generalization ability.

### Model Comparison

The three candidate architectures exhibited markedly different behaviors during training and validation:

**ResNet50** achieved the highest raw accuracy on the training set, successfully fitting the training data and demonstrating the capacity to learn the subtle discriminative features required for fine-grained animal classification. However, this high training performance came at the cost of substantial overfitting, with validation accuracy significantly lagging behind training accuracy. The large gap between training and validation performance indicates that ResNet50 learned features that were overly specific to the training set, capturing noise and idiosyncrasies rather than generalizable patterns.

**MobileNetV2** underperformed relative to the other architectures on both training and validation metrics. The reduced parameter count and efficient architecture that make MobileNetV2 attractive for deployment scenarios appear to limit its representational capacity for this challenging fine-grained classification task. The model struggled to learn sufficiently discriminative features to separate the 202 visually similar animal families, suggesting that the efficiency-accuracy tradeoff tipped too far toward efficiency for this particular problem.

**VGG19** demonstrated the most balanced performance profile, achieving competitive training accuracy while maintaining relatively good validation performance. The gap between training and validation accuracy was notably smaller than for ResNet50, suggesting that VGG19 learned more generalizable features despite the simplicity of its architectural design. This balance made VGG19 the most promising architecture for final model development.

### Final Model Performance

The tuned final model, based on the VGG19 architecture with optimized hyperparameters and the refined training strategy, achieved approximately 83.5% accuracy on the training set and 62.5% accuracy on the validation set after five training epochs. These results represent a significant improvement over the initial Grid Search results, with the gap between training and validation performance substantially reduced compared to earlier experiments.

The validation accuracy of 62.5% should be interpreted in the context of the extreme difficulty of this 202-class fine-grained classification problem with noisy, imbalanced data. Random guessing would achieve less than 0.5% accuracy, and even human experts without specialized zoological training might struggle to distinguish between many of the visually similar animal families in the dataset. The achieved performance demonstrates that the model has learned meaningful discriminative features despite the challenging data characteristics.

### Confusion Matrix Analysis

Examination of the confusion matrix revealed that the final model's predictions were homogeneously distributed across the 202 classes rather than exhibiting pathological single-class bias. This pattern indicates that the model is genuinely attempting to discriminate between different families rather than defaulting to always predicting the most common class or collapsing multiple classes into a small number of confused categories.

The absence of dominant off-diagonal patterns in the confusion matrix suggests that the model has learned family-specific features rather than relying on spurious correlations with image quality or format. However, the moderate validation accuracy indicates that many families remain difficult to distinguish, likely due to genuine visual similarity, insufficient training examples for rare families, or the presence of noisy labels and irrelevant images in the dataset.

## Installation and Usage

Setting up the project environment and running the complete experimental pipeline requires following these steps in sequence.

### Installation

Begin by installing all required dependencies using the provided requirements file. Execute the following command in your terminal within the project directory:

```bash
pip install -r requirements.txt
```

This command will install TensorFlow, Keras, NumPy, Pandas, Matplotlib, scikit-learn, and other libraries required for data processing, model training, and results visualization. Ensure that you have a compatible Python version (Python 3.8 or later is recommended) and sufficient GPU resources for training deep neural networks efficiently.

### Usage Workflow

The complete experimental pipeline should be executed in the following order to replicate the results or adapt the methodology to new data:

**Step 1: Dataset Exploration**  
Open and run the `metadata_exploration.ipynb` notebook to understand the characteristics of your dataset. This exploratory analysis will reveal the class distribution, identify potential data quality issues, and provide insights that inform subsequent preprocessing and modeling decisions. Pay particular attention to the severity of class imbalance and the prevalence of non-photographic images, as these factors significantly impact model performance.

**Step 2: Grid Search Hyperparameter Optimization**  
Execute the `colab_grid_search_routine copy.ipynb` notebook to perform systematic hyperparameter exploration for each candidate architecture. This Grid Search evaluates multiple combinations of learning rate, dense layer configuration, and dropout rate, training each configuration on a subset of the data and evaluating on a validation set. The Grid Search results identify promising hyperparameter settings that balance training performance against generalization ability. Note that this step is computationally intensive and may require several hours or days depending on available hardware resources.

**Step 3: Final Model Training**  
Run the `colab_trainig_routine.ipynb` notebook to train the final model using the best hyperparameter configuration identified during Grid Search. This notebook trains on the complete training set, monitors training and validation metrics across epochs, generates learning curves, and produces the confusion matrix for detailed performance analysis. The trained model weights are saved for subsequent evaluation on held-out test data or deployment in application scenarios.

## References

This work builds upon recent advances in animal recognition and transfer learning methodologies documented in the research literature:

Islam, M. M., et al. (2023) provide a comprehensive survey of deep learning approaches for animal species recognition, reviewing CNN architectures, transfer learning strategies, and techniques for handling class imbalance in ecological image classification tasks.

Zualkernan, I., et al. (2022) investigate the effectiveness of various pretrained CNN architectures for animal classification under real-world conditions, examining how different architectures handle noisy labels and limited training data typical of web-scraped datasets.

These references informed the architectural choices, transfer learning strategy, and evaluation methodology employed in this project, providing context for interpreting the results relative to state-of-the-art approaches in animal recognition.

---

**Project Status:** Completed as part of Deep Learning coursework.  
**Contact:** For questions or collaboration opportunities, please reach out through the GitHub repository.
