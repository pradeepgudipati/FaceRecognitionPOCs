# This Code reads a folder of images and creates a CSV file with the details of the images from the file name
# This Code is specific to a specific format and naming convention and can be modified to suit your needs
# The sample file name is abc-20210512100206-q4wny9t9yr-639773819371 -- Where abc is a constant,
# 20210512100206 is the date of onboarding, q4wny9t9yr is a unique identifier, 639773819371 is the phone number

import os
import csv

cwd = os.getcwd()
# Set the path to the folder containing the files
folder_path = os.path.join(cwd, '../Data/Selfies')
print(f"Path ---- {folder_path}")

# Create a list to store extracted data
data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a JPEG file and starts with 'abc-'
    if filename.endswith('.jpg') and filename.startswith('abc-'):
        # Split the filename to extract the required parts
        parts = filename.split('-')
        date_of_onboarding = parts[1]
        phone_number = parts[3].split('.')[0]

        # Add the details to the data list
        data.append([date_of_onboarding, phone_number, filename])

# Specify the CSV file name
csv_file = 'output.csv'

# Write the data to a CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Date of Onboarding', 'Phone Number', 'File Name'])
    writer.writerows(data)

print(f"CSV file '{csv_file}' created successfully.")
