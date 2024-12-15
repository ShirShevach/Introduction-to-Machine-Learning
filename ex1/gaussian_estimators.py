from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        if self.biased_:
            self.mu_ = np.mean(X) + 1/np.size(X)
            self.var_ = np.sum((X-self.mu_)**2)
            self.var_ *= 1/(np.size(X))
        else:
            self.mu_ = np.mean(X)
            self.var_ = np.sum((X-self.mu_)**2)
            self.var_ *= 1/(np.size(X)-1)


        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        temp = 1 / np.sqrt(2 * np.pi * self.var_)
        F = temp*(np.exp((-1/2*self.var_)*(X-self.mu_)**2))

        return F

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # raise NotImplementedError()
        # a = (-(np.shape(X)[0]/2))*np.log(2*np.pi*sigma)
        # b = (1/(2*sigma))*((X-mu)**2)
        # return a - b
        n = np.shape(X)[0]
        a = -(n/2) * np.log(2 * np.pi)
        b = -(n/2) * np.log(sigma**2)
        c1 = -(1/2 * sigma**2)
        c2 = 0
        for x in X:
            c2 += (x-mu)**2
        return a + b + (c1*c2)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        # raise NotImplementedError()
        temp = np.zeros(np.shape(X)[1])
        for j in range(np.size(X, axis=1)):
            sum = 0
            for i in range(np.size(X, axis=0)):
                sum += X[i][j]
            temp[j] = sum / np.size(X, axis=0)

        self.mu_ = temp
        self.cov_ = np.cov(X.T, bias=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        # raise NotImplementedError()
        F = np.zeros(np.shape(X)[0])
        d = np.size(X, axis=1)
        det_sigma = np.linalg.det(self.cov_)
        cov_inv = np.linalg.inv(self.cov_)
        temp = 1 / np.sqrt(((2 * np.pi) ** d) * det_sigma)
        for j in range(np.size(X, axis=0)):
            b = (X[j, :] - self.mu_)
            inside = -0.5 * np.linalg.multi_dot((b, cov_inv, b))
            a = np.exp(inside)
            F[j] = temp * a
        return F


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        # raise NotImplementedError()
        cov_trans = np.transpose(cov)
        a = -((np.shape(X)[0]*np.shape(X)[1])/2)*np.log(2*np.pi)
        b = (np.shape(X)[0]/2)*np.log(np.linalg.det(cov_trans))

        c = 0
        for j in range(np.shape(X)[0]):
            c += (X[j]-mu).T@cov_trans@(X[j]-mu)
        c *= -0.5

        return a+b+c