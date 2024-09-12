import os

# Parameters for the combinations
commodities = ['bawang_merah', 'cabai_merah_besar', 'cabai_merah_keriting', 'cabai_rawit_hijau', 'cabai_rawit_merah']
neurons = [16, 32, 64]
epochs = [100, 150]
batch_sizes = [32]

# Path where you want to create the folders (use your provided base path)
base_path = r"E:\Kuliah\Semester 8\Skripsi\ANALISA HASIL\result_saved_code"  # Use 'r' to handle backslashes in Windows paths

# Initialize the order number
order_number = 1

# Loop through each combination of parameters and create folders
for commodity in commodities:
    for epoch in epochs:
        for neuron in neurons:
            for batch_size in batch_sizes:
                # Create folder name based on format [order_number]_commodityname_epoch_neuron_batchsize
                folder_name = f"{order_number}_{commodity}_epoch{epoch}_neuron{neuron}_batchsize{batch_size}"
                folder_path = os.path.join(base_path, folder_name)
                
                # Create the folder
                os.makedirs(folder_path, exist_ok=True)
                
                # Increment order number for each new folder
                order_number += 1

print(f"{order_number-1} folders created successfully.")
