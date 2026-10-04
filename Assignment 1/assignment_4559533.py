# No external libraries are allowed to be imported in this file
import sklearn
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import random


# 1. COSINE SIMILARITY:
def similarity_matrix(matrix, k=5, axis=0):
    """
    This function should contain the code to compute the cosine similarity
    (according to the formula seen at the lecture) between users (axis=0) or 
    items (axis=1) and return a dictionary where each key represents a user
    (or item) and the value is a list of the top k most similar users or items,
    along with their similarity scores.
    
    Args:
        matrix (pd.DataFrame) : user-item rating matrix (df)
        k (int): number of top k similarity rankings to return for each \
                    entity (default=5)
        axis (int): 0: calculate similarity scores between users \
                        (rows of the matrix), 
                    1: claculate similarity scores between items \
                        (columns of the matrix)
    
    Returns:
        similarity_dict (dictionary): dictionary where the keys are users 
        (or items) and the values are lists of tuples containing the most 
        similar users (or items) along with their similarity scores.

    Note that is NOT allowed to authomatically compute cosine similarity using
    an apposite function from any package, the computation should follow the 
    formula that has been discussed during the lecture and that can be found in
    the slides.

    Note that is allowed to convert the DataFrame into a Numpy array for 
    faster computation.
    """
    similarity_dict= {}
    # TO DO: Handle the absence of ratings (missing values in the matrix)
    
    
    # TO DO: If axis is 1, what do you need to do to calculate the similarity 
    # between items (columns)
    
    if axis==1:
        matrix = matrix.T

    """Above segment deals with the axis argument, transposes the matrix if we want items instead of users"""
    # TO DO: loop through each couple of entities to calculate their cosine 
    # similarity and store these results
    
    labels = matrix.index
    values = matrix.to_numpy()
    n = values.shape[0]
    num = 0
    denom = 0
    cossim = 0
    simmatrix = np.full((n, n), np.nan)
    
    for i in range(n):
        row1 = values[i]
        mask1 = ~np.isnan(row1)
        for j in range(n):
            if i == j:
                continue
            row2 = values[j]
            mask2 = ~np.isnan(row2)
            mask = mask1 & mask2
            row1common = row1[mask]
            row2common = row2[mask]
            num = sum(row1common * row2common)
            denom = (sum(x*x for x in row1common) * sum(y*y for y in row2common)) ** 0.5
            cossim = np.nan if denom == 0 else num / denom
            simmatrix[i, j] = cossim  

    """Above code computes similarities with a nested for loop for each pair of rows, then places this cosine similarity
       in the correct index."""
        
    
    # TO DO: sort the similarity scores for each entity and add the top k most 
    # similar entities to the similarity_dict
        
    for i in range(n):
        row = simmatrix[i]
        valid = ~np.isnan(row)
        idx_valid = np.where(valid)[0]
        scores_valid = row[valid]
        order = np.argsort(-scores_valid)  # descending
        top_idx = idx_valid[order][:k]
        top_scores = scores_valid[order][:k]
        similarity_dict[labels[i]] = [(labels[t], s) for t, s in zip(top_idx, top_scores)]

    return similarity_dict


# 2. COLLABORATIVE FILTERING
def user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement user-based collaborative
    filtering, returning the predicted rate associated to a target user-movie
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        user_similarity (dict): dictonary containing user similarities, \
            obtained using the similarity_matrix function (axis=0)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to user-based \
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target user
    topsim = user_similarity[user_id][:k]

    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0
    for (x,y) in topsim:
        userrating = user_item_matrix.loc[x, movie_id]
        if pd.isna(userrating):
            continue
        numerator += y * userrating
        denominator += y

      


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating


def item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5):
    """
    This function should contain the code to implement item-based collaborative
    filtering, returning the predicted rate associated to a target user-movie 
    pair.

    Args:
        user_id (int): target user ID
        movie_id (int): target movie ID
        item_similarity (dict): dictonary containing item similarities, \
            obtained using the similarity_matrix function (axis=1)
        user_item_matrix (pd.DataFrame): user-item rating matrix (df)
        k (int): number of top k most similar users to consider in the \
            computation (default=5)

    Returns:
        predicted_rating (float): predicted rating according to item-based 
        collaborative filtering
    """
    # TO DO: retrieve the topk most similar users for the target user
    topsim = item_similarity[movie_id][:k]
    user_item_matrix = user_item_matrix.T

    # TO DO: implement user-based collaborative filtering according to the 
    # formula discussed during the lecture (reported in the PDF attached to 
    # the assignment)
    numerator = 0  
    denominator = 0
    for (x,y) in topsim:
        movierating = user_item_matrix.loc[x, user_id]
        if pd.isna(movierating):
            continue
        numerator += y * movierating
        denominator += y

      


    if denominator == 0:
        return np.nan  # no similar users or no valid ratings, NaN is returned.

    predicted_rating = numerator / denominator

    return predicted_rating

# 3. MATRIX FACTORIZATION
def matrix_factorization(
        utility_matrix: np.ndarray,
        feature_dimension=2,
        learning_rate=0.001,
        regularization=0.02,
        n_steps=2000
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function should contain the code to implement matrix factorisation
    using the Gradient Descent with Regularization method (according to the psuedo code
    seen at the lecture), returning the user and item matrices.

    Args:
        utility_matrix (np.ndarray): user-item rating matrix
        feature_dimension (int): number of latent features (default=2)
        learning_rate (float): learning rate for gradient descent \
            (default=0.001)
        regularization (float): regularization parameter (default=0.02)
        n_steps (int): number of iterations for gradient descent \
            (default=2000)

    Returns:
        user_matrix (np.ndarray): user matrix
        item_matrix (np.ndarray): item matrix
    """

    user_matrix = np.random.rand(utility_matrix.shape[0], feature_dimension)
    item_matrix = np.random.rand(utility_matrix.shape[1], feature_dimension)

    known_indices = [
        (i, j)
        for i in range(utility_matrix.shape[0])
        for j in range(utility_matrix.shape[1])
        if not np.isnan(utility_matrix[i, j])
    ]
    for step in range(n_steps):
        random.shuffle(known_indices)  # stochastic: shuffle order each pass
        for i, j in known_indices:
            rating = utility_matrix[i, j]

            prediction = np.dot(user_matrix[i], item_matrix[j])
            err = rating - prediction

            uuser = user_matrix[i].copy()
            vitem = item_matrix[j].copy()

            user_matrix[i] = uuser + learning_rate * (err * vitem - regularization * uuser)
            item_matrix[j] = vitem + learning_rate * (err * uuser - regularization * vitem)

    return user_matrix, item_matrix

if __name__ == "__main__":
    path = r"C:\Leiden Courses\Advances in Data Mining\Practical\Assignment 1\u.data"      
    df = pd.read_table(path, sep="\t", names=[
        "UserID", "MovieID", "Rating", "Timestamp"
    ])
    df = df.pivot_table(
        index = 'UserID', 
        columns = 'MovieID', 
        values = 'Rating'
    )

    # You can use this section for testing the similarity_matrix function: 
    # Return the top 5 most similar users to user 3:
    user_similarity_matrix = similarity_matrix(df, k=5, axis=0)
    print(user_similarity_matrix.get(3,[]))

    # Return the top 5 most similar items to item 10:
    item_similarity_matrix = similarity_matrix(df, k=5, axis=1)
    print(item_similarity_matrix.get(10,[]))

    
    # You can use this section for testing the user_based_cf and the 
    # item_based_cf functions: Return the predicted ratings assigned by user 
    # 13 to movie 100:
    user_id = 13 
    movie_id = 100

    u_predicted_rating = user_based_cf(
        user_id, 
        movie_id, 
        user_similarity_matrix, 
        user_item_matrix = df,
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to user-based collaborative filtering is: "
        f"{u_predicted_rating:.2f}"
    )

    i_predicted_rating = item_based_cf(
        user_id,
        movie_id, 
        item_similarity_matrix,
        user_item_matrix = df, 
        k=5
    )
    print(
        f"predicted user {user_id} rating for movie {movie_id}, "
        f"according to item-based collaborative filtering is: "
        f"{i_predicted_rating:.2f}"
    )

    utility_matrix = np.array([
        [5, 2, 4, 4, 3],
        [3, 1, 2, 4, 1],
        [2, np.nan, 3, 1, 4],
        [2, 5, 4, 3, 5],
        [4, 4, 5, 4, np.nan],
    ])
    user_matrix, item_matrix = matrix_factorization(
        utility_matrix, learning_rate=0.001, n_steps=5000
    )

    print("Current guess:\n", np.dot(user_matrix, item_matrix.T))
