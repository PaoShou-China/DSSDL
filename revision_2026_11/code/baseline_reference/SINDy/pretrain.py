from datasets import dataset_dict
import datasets as datasets_module
from SINDy import SINDyRegressor

# --- Main Execution ---
if __name__ == "__main__":
    # Initialize SINDyRegressor with dataset
    system_name = getattr(datasets_module, "CURRENT_SYSTEM", "sindy")
    regressor = SINDyRegressor(dataset_dict=dataset_dict, system_name=system_name)

    # Train the model
    regressor.train()

    # Evaluate RMSE on the query split
    regressor.evaluate_on_query()

    # Save the trained model
    regressor.save()
