import pickle
import os
import glob

def merge_pickles(target_dir, output_file):
    merged_data = {}

    for file in glob.glob(os.path.join(target_dir, "data_*.pkl")):
        with open(file, 'rb') as f:
            try:
                data = pickle.load(f)
                merged_data.update(data)
            except (EOFError, pickle.UnpicklingError):
                print(f"Warning: Skipping corrupted file {file}")
                continue  # Skip deletion of corrupted files

        # Delete the successfully processed pickle file
        os.remove(file)

    # Save merged data to a final pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(merged_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Merging complete. Data saved in {output_file}. Individual files deleted.")

# Example usage:
merge_pickles("/mnt/users/kotssvasiliou/ED/new_dir_66ec702c", "final_data_2.pkl")
