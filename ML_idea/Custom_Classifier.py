from sklearn.base import BaseEstimator, ClassifierMixin


class ActivitiesPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, model='Single'):
        """
        Called when initializing the classifier
        """

        self.model = model
