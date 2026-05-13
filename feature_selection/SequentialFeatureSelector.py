import numpy as np
from sklearn.model_selection import cross_val_score , StratifiedKFold ,KFold
from sklearn.base import is_classifier

class SequentialFeatureSelector:
    """
    Constructor for the Sequential Feature Selector

    Parameters:
    - estimator: the machine learning model (LogisticRegression, DecisionTree,etc.)
    - n_features_to_select: number of features we want to keep
    - direction: 'forward' (add features) or 'backward' (remove features)
    - scoring: evaluation metric used in cross-validation
    """
    def __init__(self, estimator, n_features_to_select, direction="forward", scoring="accuracy"):
          self.estimator = estimator
          self.n_features_to_select = n_features_to_select
          self.direction = direction
          self.scoring = scoring
          self.selected_features = None

    def fit(self, X, y):
          """
          Select the best features based on the chosen method (forward/backward)
          Parameters:
              - X: feature matrix (numpy array)
              - y: target vector
          """
          n_features = X.shape[1]
          self.n_total_features = n_features

          # Validation
          if self.direction not in ["forward", "backward"]:
            raise ValueError("direction must be 'forward' or 'backward'")

          if self.n_features_to_select > n_features:
            raise ValueError("n_features_to_select cannot be greater than total features")

          if self.n_features_to_select <= 0:
            raise ValueError("n_features_to_select must be > 0")

          # Auto scoring
          if self.scoring is None:
            if is_classifier(self.estimator):
               self.scoring = "accuracy"

            else: self.scoring = "r2"

          # Choose CV strategy
          if is_classifier(self.estimator):
             cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

          else: cv = KFold(n_splits=5, shuffle=True, random_state=42)

          if self.direction == "forward":
              selected = set()

              while len(selected) < self.n_features_to_select:
                  best_score = -np.inf
                  best_feature = None

                  for i in range(n_features):
                      if i not in selected:
                          current_features = list(selected) + [i]
                          X_subset = X[:, current_features]
                          score = cross_val_score(self.estimator, X_subset, y,cv=cv, scoring=self.scoring).mean()
                          if score > best_score:
                              best_score = score
                              best_feature = i

                  selected.add(best_feature)

          elif self.direction == "backward":
              selected = set(range(n_features))

              while len(selected) > self.n_features_to_select:
                  best_score = -np.inf
                  bad_feature = None

                  for i in selected:
                      current_features = list(selected - {i})
                      X_subset = X[:, current_features]
                      score = cross_val_score(self.estimator, X_subset, y,cv=cv, scoring=self.scoring).mean()
                      if score > best_score:
                          best_score = score
                          bad_feature = i

                  selected.remove(bad_feature)

          self.selected_features = sorted(selected)
          return self

    def transform(self, X):
        """
        Reduce dataset to selected features only
        Parameters:
        - X: original feature matrix

        Returns:
        - Transformed matrix with selected features
        """
        if self.selected_features is None:
           raise ValueError("You must call fit() before transform()")

        return X[:, self.selected_features]

    def fit_transform(self, X, y):
          """
          Fit the selector and transform data in one step
          """
          self.fit(X, y)
          return self.transform(X)

    def get_support(self):
        """
        Return a boolean mask of selected features
        """
        if self.selected_features is None:
          raise ValueError("You must call fit() before get_support()")

        mask = np.zeros(self.n_total_features, dtype=bool)
        mask[self.selected_features] = True
        return mask