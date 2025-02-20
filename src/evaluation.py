import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error,r2_score 
import numpy as np

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass

class MSE(Evaluation):

    def calculate_scores(self,y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging. info ("Calculating R2 Score")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}". format (e) )
            raise e
        
class RMSE(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error (y_true, y_pred, squared=False)
            logging. info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error ("Error in calculating RMSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:            
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
    
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e
        