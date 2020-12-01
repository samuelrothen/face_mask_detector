import os

# Small tool for batch-renaming Files

path = '../data/no_mask/'
files = os.listdir(path)

for i, file in enumerate(files):
    file_name, ext = os.path.splitext(file)
    old_file = f'{path}{file}'
    new_filename = f'no_mask_{i+1}{ext}'
    os.rename(old_file, new_filename)
