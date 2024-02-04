# Book Recommendation System using PCA, K-Means and K-Nearest Neighbors

- Article on Medium: https://medium.com/@samir.silva12342/making-a-book-recommendation-system-with-pca-k-means-and-k-nearest-neighbors-d23f0a31aaf2

## Idea

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

## Results

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

- Recommendation Algorithm:

<p align="center">
    <img width="700" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/algorithm.png" alt="Material Bread logo">
<p>

- Book Recommendations:

<p align="center">
    <img width="700" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/recommendations.png" alt="Material Bread logo">
<p>
