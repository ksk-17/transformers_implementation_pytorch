import unicodedata
from tqdm import tqdm
import re

dataset_path = "C:/Users/hp/data-science-from-scratch/transformer_impl/datasets"
training_folders = ["training_commoncrawl/commoncrawl"]
test_folders = ["test/newstest2014-deen-src"]
cleaned_folder = "cleaned_dataset"
translation_task = "de-en"
languages = ["de", "en"]

# Normalize the text to Unicode NFKD form, format quotes and remove other characters
def clean_text(text):
    text =  unicodedata.normalize('NFKD', text.lower())
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = re.sub('[^A-Za-zäöüß ]+', '', text)
    return text

def prepare_dataset(train_folders, mode = "train"):
    for train_folder in train_folders:
        file_path_1 = dataset_path + "/" + train_folder + "." + translation_task + "." + languages[0]
        file_path_2 = dataset_path + "/" + train_folder + "." + translation_task + "." + languages[1]
        print('started reading file')

        en_lines = []
        de_lines = []

        with open(file_path_1, 'r', encoding='utf-8') as file1, open(file_path_2, 'r', encoding='utf-8') as file2:
            file1_lines = file1.readlines()
            file2_lines = file2.readlines()
            total_length = len(file1_lines)

            with tqdm(total=total_length, desc="Progress", unit="char") as pbar:
                for i in range(total_length):
                    en_lines.append(clean_text(file1_lines[i]))
                    de_lines.append(clean_text(file2_lines[i]))
                    # Update progress bar
                    pbar.update(1)

        return en_lines, de_lines

def train_data():
    print("started processing training data")
    output = prepare_dataset(train_folders=training_folders, mode="train")
    print("completed processing training data")
    return output













