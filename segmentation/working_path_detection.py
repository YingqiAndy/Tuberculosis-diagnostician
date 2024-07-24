import os
#The purpose of this py file is to make it easier for you to determine the execution path of the various parts of the software
# Output current directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Switch to the new directory
new_directory = "../data/1000_external_data/image"
os.chdir(new_directory)

# Output the current working directory again and verify whether the switchover is successful
updated_directory = os.getcwd()
print("Updated working directory:", updated_directory)