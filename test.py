import os

def create_folder_and_file(base_path, folder_name, file_name):
    # Construct the full folder path
    #folder_path = os.path.join(base_path, folder_name)
    
    # Create the folder if it doesn't exist
    os.makedirs("/work3/arnaou/newfolder", exist_ok=True)

    # Construct the file path
    #file_path = os.path.join(folder_path, file_name)

    # Write "hello" to the file
    with open("/work3/arnaou/newfolder/example.txt', "w") as file:
        file.write("hello")

    #print(f"Folder and file created at {file_path}")

# Example usage
# Replace '/absolute/path/to/existing/folder' with your desired absolute base path
create_folder_and_file("/work3/arnaou", "new_folder", "example.txt")