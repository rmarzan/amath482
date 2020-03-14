# AMATH 482: Computational Methods for Data Analysis
Hello there! My name is Rosemichelle Marzan (UW Bioengineering, 2020) and this is my repository for my Winter 2020 AMATH 482 course. Here, you will find brief project reports I have written demonstrating what I have learned from the course with regards to filtering methods, spectral analysis, principal component analysis, linear discriminant analysis, and machine learning classification. The official course description is as follows:

> Exploratory and objective data analysis methods applied to the physical, engineering, and biological sciences. Brief review of statistical methods and their computational implementation for studying time series analysis, spectral analysis, filtering methods, principal component analysis, orthogonal mode decomposition, and image processing and compression. *(University of Washington Dept. of Applied Mathematics Course Catalog)*

## Projects Directory
* [HW1 - Ultrasound Problem](https://github.com/rmarzan/amath482/tree/master/HW1%20-%20Ultrasound%20Problem) 
  * This report demonstrates how the fast Fourier transform and **Gaussian-based spectral filtering** can be applied to a hypothetical, yet relatively realistic, scenario. The proposed solution denoises the given noisy dataset using the averaging method to determine the object's frequency signature, which is then used as the center frequency in the applied Gaussian filter. Applying the inverse fast Fourier transform to the filtered signal allows the object to be tracked spatially over time.
* [HW2 - Gabor Transforms](https://github.com/rmarzan/amath482/tree/master/HW2%20-%20Gabor%20Transforms) 
  * This report demonstrates how **Gabor transforms** are used to analyze different audio signals. In Part I, we explore the effects of modulating the Gabor transform for filtering anaudio signal. In Part II, we reproduce the music scores for two instruments based on the spectrograms produced by Gabor filtering.
* [HW3 - Principal Component Analysis](https://github.com/rmarzan/amath482/tree/master/HW3%20-%20Principal%20Component%20Analysis) 
  * The positional coordinates of a moving object filmed using three cameras in four different experiments were obtained with a computational algorithm. **Singular value decomposition** was then performed on these data sets, the results of which were then interpreted with principal component analysis. This report demonstrates the practical usefulness of principal component analysis for examining real-world data which may contain redundancies in information and noise, and how these factors influence the accuracy of the algorithm.
* [HW4 - Music Classification](https://github.com/rmarzan/amath482/tree/master/HW4%20-%20Music%20Classification) 
  * Principal component analysis (PCA) and **linear discriminant analysis (LDA)** were used to create a machine learning algorithm for classifying 5-second audio clips. Three different classifiers were used to classify songs from three artists of different genres, three artists from the same genre, and many artists from three genres. The classifiers were evaluated on their success rate. This report discussing the effects of training data variation and the number of PCA modes ("features") on classifier performance.
* [HW5 - Neural Networks](https://github.com/rmarzan/amath482/tree/master/HW5%20-%20Neural%20Networks) 
  * A fully connected neural network and a convolutional neural network were trained to classify items from the Fashion-MNIST dataset. The fully connected neural network correctly identified 88.06% of the items and the convolutional neural network correctly classified 91.68% of the items in the test set. The goal of this study is to experimentally adjust each network's hyperparameters to gain a basic understanding of their effect on the network's validation performance and to find the optimum parameters for training the final model.
