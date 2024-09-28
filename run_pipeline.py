from pipelines.training_pipeline import training_pipeline
# from zenml.client import Client

if __name__ == "__main__":
    # print('here',Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="`./data/olist_customers_dataset.csv`")

    