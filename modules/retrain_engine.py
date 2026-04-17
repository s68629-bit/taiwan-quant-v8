import datetime
from config.settings import RETRAIN_DAYS
def need_retrain(last_train_date):
    return (datetime.date.today() - last_train_date).days >= RETRAIN_DAYS
