# Book Recommendation System

- Article on Medium: IN PROGRESS...

## Idea

We will make a book recommendation system based in the user's last read books. Our final objective is to recommend 10 books for each person. For this, we will first clusterize the books using K-Means algorithm after making PCA (Principal Component Analysis). This will be made by considering some characteristics:

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

Then, we will recommend numbers of books of the clusters which have books the person already read. Using bootstrap, we will select randomly the next read book to consider to select another book from the same cluster as it. This selection will be made using K-Nearest Neighbors: we will calculate nearest neighbors from the same cluster and then select one of them to compose the recommendations - the additional condition is that the book must not have been recommended yet. The number of nearest neighbors will be determined by the length of the already read books, to guarantee that we will have sufficient neighbors to look for a new book to recommend.

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
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/algorithm.png" alt="Material Bread logo">
<p>

- Book Recommendations:

<p align="center">
    <img width="600" src="https://github.com/Samirnunes/data-science/blob/main/machine_learning/book_recommendation/images/recommendations.png" alt="Material Bread logo">
<p>

