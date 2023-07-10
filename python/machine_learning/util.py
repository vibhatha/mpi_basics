import os
import imageio
import re


def sort_strings_with_suffix(strings):
    def extract_suffix(string):
        # Extract the suffix using regular expression
        suffix = re.findall(r'\d+', string)
        
        if suffix:
            return int(suffix[0])
        else:
            return string

    return sorted(strings, key=lambda s: extract_suffix(s))


def generate_gif(path, file_prefix, output_path, output_file, duration):
    frames = []
    list_files = os.listdir(path)
    list_files_sorted = sort_strings_with_suffix(list_files)
    for file in list_files_sorted:
        if file_prefix in file:
            print(file)
            image = imageio.v2.imread(os.path.join(path,file)) 
            frames.append(image)
    imageio.mimsave(os.path.join(output_path, output_file),
                frames,          
                duration = duration)