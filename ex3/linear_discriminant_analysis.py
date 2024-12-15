from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
from ...learners import gaussian_estimators


class LDA(BaseEstimator):
# class LDA():
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.classes_, counts = np.unique(y, return_counts=True)

        self.pi_ = counts / n_samples  # todo maybe efshar lehorid

        self.mu_ = np.zeros((len(counts), n_features))
        self.cov_ = np.zeros((n_features, n_features))
        for i, k in enumerate(self.classes_):
            x_by_k = X[y == k]
            S1 = np.sum(x_by_k, axis=0)
            self.mu_[i] = S1 / len(x_by_k)

            tmp = x_by_k - self.mu_[i]
            M = tmp.T @ tmp
            self.cov_ += M
        self.cov_ *= 1 / (n_samples-self.classes_.shape[0])  # m-k

        self._cov_inv = inv(self.cov_)
        print(self._cov_inv)


    def calc_A_B(self):
        a = []
        b = []
        for k in range(self.pi_.shape[0]):
            a.append(self._cov_inv @ self.mu_[k])
            b.append(np.log(self.pi_[k]) - 0.5 * self.mu_[k].T @ self._cov_inv @ self.mu_[k])
        return a, b

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y = np.zeros(X.shape[0])
        candidate_y_for_x = np.zeros(len(self.classes_))
        AB_per_k = np.zeros((len(self.classes_), 3))
        for j, k in enumerate(self.classes_):
            A = self._cov_inv @ self.mu_[j]
            B = np.log(self.pi_[j]) - 0.5 * (self.mu_[j].T @ self._cov_inv @ self.mu_[j])
            AB_per_k[j][0:2], AB_per_k[j][2] = A, B

        for i, x in enumerate(X):
            for j, k in enumerate(self.classes_):
                candidate_y_for_x[j] = AB_per_k[j][0:2].T@x + AB_per_k[j][2]
            y[i] = self.classes_[np.argmax(candidate_y_for_x)]

        return y


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        a = 1/np.sqrt(((2*np.pi)**X.shape[1]) @ det(self.cov_))
        x_mu = X - self.mu_
        b = np.exp(-0.5*x_mu.T @ self._cov_inv @ x_mu)
        return self.pi_ @ a @ b

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)

# if __name__ == '__main__':
#     X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#     y = np.array([1, 1, 1, 2, 2, 2])
#     A = LDA()
#     A._fit(X,y)
