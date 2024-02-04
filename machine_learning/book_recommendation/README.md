# Book Recommendation System
using PCA, K-Means and K-Nearest Neighbors

- Article on Medium: https://medium.com/@samir.silva12342/making-a-book-recommendation-system-with-pca-k-means-and-k-nearest-neighbors-d23f0a31aaf2

## Idea

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/book_recommendation_system.png" alt="Material Bread logo">
<p>

We will make a book recommendation system based in the user's last read books. Our final objective is to recommend 10 books for each person. For this, we first make PCA (Principal Component Analysks) and then clusterize the books using K-Means. The clusters are created considering some characteristics:

- Year of Publication
- Mean Age of Readers by Book
- Mean Age of Readers by Author
- Mean Age of Readers by Publisher
- Mean Rating by Book
- Mean Rating by Author
- Mean Rating by Publisher
- Top Location by Book (Count Encoded) - where the book has been more read.
- Book Contribution to Top Location - what is the fraction of readings the book contributed to in its top place.
- Book Author (Count Encoded)
- Publisher (Count Encoded)
- Book Appearances (Count encoding of ISBN)

After that, we recommend books from the clusters which have a book that the person has already read. In this context, using bootstrap, we randomly select the next book that was read so we can select another book from the same cluster as it. This selection is done with K-Nearest Neighbors: we calculate the nearest neighbors from the same cluster and then select one of them to compose the recommendations - the additional condition is that the book must not have been recommended yet. The number of nearest neighbors will be determined by the length of the already read books, to guarantee that we will have sufficient neighbors to look for a new book to recommend.

## Data

Books Dataset From Kaggle: https://www.kaggle.com/datasets/saurabhbagchi/books-dataset

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/data.png" alt="Material Bread logo">
<p>

## Results

### Code

The following files compose the project:

- `system.ipynb` notebook containing the system development from the start. It shows step-by-step how the recommendation system was created.

- `recommendations.ipynb` notebook which uses the final implementation of the recommendation system. The final implementation encapsulates the system into the `BookRecommendationAlgorithm` class.

- The final implementation is composed by the following files:
    - `book_recommendation_algorithm.py`: contains the `BookRecommendationAlgorithm` class, which holds the final implemention of the recommendation algorithm.
    - `book_recommendation_preprocessor.py`:  contains the `BookRecommendationPreprocessor` class, which makes the data preprocessing steps.
    - `book_main_step_preprocessor.py`: contains the `BookMainStepPreprocessor` class, which makes the preprocessing of the main dataframe (the main dataframe contains all the informations from which the other ones derive).
    - `book_recommendation_k_means.py`: contains the `BookRecommendationKMeans` class, which implements the k-means algorithm's steps to use for the clusters generation.
    - `book_recommendation_k_nearest.py`: contains the `BookRecommendationKNearest` class, which implements the k-nearest neighbors algorithm based on a within cluster perspective.
    - `utils.py`: used to make the data import steps easier.
      
### Images

- Initial Correlations:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/correlations_initial.png" alt="Material Bread logo">
<p>

- PCA (Principal Component Analysis):

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/pca.png" alt="Material Bread logo">
<p>

- Final Correlations after PCA:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/correlations_final.png" alt="Material Bread logo">
<p>

- Elbow Method to Determine Number of Clusters in K-Means:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/elbow.png" alt="Material Bread logo">
<p>

- Book Recommendations:

<p align="center">
    <img width="800" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/recommendations.png" alt="Material Bread logo">
<p>
