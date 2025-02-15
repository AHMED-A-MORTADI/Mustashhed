# MODEL_NAME = "arabert_base"
# MODEL_NAME = "arabert_large"
MODEL_NAME = "Arabic_KW_Mdel"
# MODEL_NAME = "KW_Model_without_SW"


REMOVE_SENTENCE_IF_BECOME_SHORT_AFTER_REMOVING_STOPWORDS = False
REMOVE_SENTENCE_IF_BECOME_SHORT_AFTER_REMOVING_STOPWORDS_N = 3
MAX_NUM_OF_EXAMPLES_PER_WORD = 1000

EXAMPLE_THRESHOLD = 0.35
REMOVE_STOPWORDS_FROM_MEANING = True
REMOVE_WORD_FROM_MEANING = True
REMOVE_EXCLUDED_WORDS_FROM_MEANING = True

REMOVE_SIMILAR_EXAMPLES_THRESHOLD = 0.000005
TEXT_WINDOW_SIZE = 9
# DISPLAY_WAY = 1 # DISPLAY_EXAMPLES_GROUPED_BY_WORDS_AND_SORT_WORDS_BASED_ON_AVERAGE_AND_SORT_EXAMPLES_WHIN_GROUPES_BASED_ON_SCORE
DISPLAY_WAY = 2 # DISPLAY_EXAMPLES_SORTED_BY_SCORE_AND_GROUP_CONSECUTIVE_PART_BY_WORD
GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2 = False

API_KEY = ""

SUPPORTED_EXAMPLES_TYPES = ["news","quraan","poetry","hadith"]
# SUPPORTED_EXAMPLES_TYPES = ["poetry","hadith","quraan"]


import os
import time
from collections import defaultdict
import re
import requests
import csv
os.environ['CAMELTOOLS_DATA'] = r'static/resources/camel_tools'

from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModel
import pickle
import torch
import faiss
import numpy as np
from flask import Flask
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.reinflector import Reinflector
from itertools import product
from camel_tools.utils.dediac import dediac_ar
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


EXAMPLES_TYPES_REQUIRE_METADATA = ["quraan","hadith"]

Excluded_Words_From_Meanings = ["فلان", "فلانا", "لفلان", "بفلان", "شيء", "الشيء", "أمر", "الأمر", "كذا", "بكذا", "فعل", "الفعل", "ونحوه", "ونحوها", "غيره", "وغيره", "وغيرها"]

class MustashhedApp:
    def __init__(self, name):
        self.app = Flask(name)
        self.setup_routes()
        self.mode = "light"
        self.word = ""
        self.meaning = ""
        self.word_type = "Noun"
        self.resource_type = "all"
        self.examples = []
        self.list_of_meanings_dicts = []
        self.lemma_custom_checkbox = True
        self.setup = 2

        self.sentences = self.get_all_sentences()
        self.inverted_indices = self.get_all_inverted_indices()
        self.embeddings = self.get_all_embeddings()
        self.types_metadata = self.get_types_metadata()
        self.model, self.tokenizer = self.get_model_and_tokenizer()
        self.arabic_stopwords = self.load_arabic_stopwords()
        self.reinflector = Reinflector(MorphologyDB.builtin_db(flags='r'))
        print("CAMLTool's setup Completed")

    def setup_routes(self):
        self.app.add_url_rule('/', 'index', self.index, methods=['GET', 'POST'])
        self.app.add_url_rule('/get_examples_setup_1', 'get_examples_setup_1', self.get_examples_setup_1, methods=['GET', 'POST'])
        self.app.add_url_rule('/get_examples_setup_2', 'get_examples_setup_2', self.get_examples_setup_2, methods=['GET', 'POST'])
        self.app.add_url_rule('/change_the_setup', 'change_the_setup', self.change_the_setup, methods=['GET', 'POST'])
        self.app.add_url_rule('/set_mode', 'set_mode', self.set_mode, methods=['GET', 'POST'])
        self.app.add_url_rule('/contribute_in_alriyadh_dictionary_add', 'contribute_in_alriyadh_dictionary_add', self.contribute_in_alriyadh_dictionary_add, methods=['GET', 'POST'])
        self.app.add_url_rule('/contribute_in_alriyadh_dictionary_report', 'contribute_in_alriyadh_dictionary_report', self.contribute_in_alriyadh_dictionary_report, methods=['GET', 'POST'])


    def change_the_setup(self):
        try:
            self.setup = request.form.get('destination')
            return render_template(f'setup_{self.setup}.html',examples=[], word="", meaning="", mode=self.mode, word_type="Noun", resource_type = self.resource_type,GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)

    def index(self):
        mode = request.form.get('mode')

        if mode == None:
            if self.mode != None:
                mode = self.mode
            else:
                mode = "light"
        if not self.setup:
            self.setup = 1
        return render_template(f'setup_{self.setup}.html',examples=[], word="", meaning="", mode=mode, word_type="Noun", resource_type = self.resource_type)

    def get_examples_setup_1(self):
        try:
            self.word = request.form.get('word')
            self.meaning = request.form.get('meaning')
            self.resource_type = request.form.get('resource_type')
            self.word_type = request.form['type-select']

            try:
                self.examples = self.get_examples_with_faiss(self.word, self.meaning, self.word_type, self.resource_type)
            except:
                print("Exception in get_examples_setup_1()")
                self.write_to_logs("Exception in get_examples_setup_1()")

            if len(self.examples) == 0 :
                self.examples = [("",["لم نجد أمثلة!"])]
            return render_template(f'setup_{self.setup}.html', examples=self.examples, word=self.word,meaning=self.meaning, word_type= self.word_type,resource_type = self.resource_type , mode=self.mode,GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)

    def get_examples_setup_2(self):
        # try:
            self.word = request.form.get('word')
            self.meaning = request.form.get('meaning')
            self.resource_type = request.form.get('resource_type')

            examples_list = []
            self.examples= examples_list

            if self.meaning == 'لعرض المعاني: قم بكتابة الكلمة ثم اضغط على "استرجاع المعاني"' or self.meaning == None:
                self.list_of_meanings_dicts = self.get_list_of_meanings_dicts_for_word(self.word)
                return render_template(f'setup_{self.setup}.html',list_of_meanings_dicts=self.list_of_meanings_dicts, examples=None, word=self.word,meaning=self.meaning, word_type= self.word_type,resource_type = self.resource_type , mode=self.mode,GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)

            else:
                # try:
                    self.meaning = request.form.get('meaning')
                    for item in self.list_of_meanings_dicts:
                        if self.meaning == item['meaning']: # becuase the meaning in the UI is consists from word_with_diacr+ space+meaning
                            meaning_dict = item
                            type = item['pos']
                            word_with_diacr = item['word_with_diacr']
                            if "V" in type:
                                self.word_type = "Verb"
                            elif "N" in type:
                                self.word_type = "Noun"
                            elif "R" in type:
                                self.word_type = "Preposition"
                            else:
                                self.word_type = "Noun"
                            break

                    self.examples = self.get_examples_with_faiss(self.word, self.meaning, self.word_type, self.resource_type)
                    if len(self.examples) == 0 :
                        self.examples = [("",["لم نجد أمثلة!"])]
                # except:
                #     self.write_to_logs("Exception in get_examples_setup_2()")

            return render_template(f'setup_{self.setup}.html',selected_meaning=meaning_dict,list_of_meanings_dicts=self.list_of_meanings_dicts, examples=self.examples, word=self.word,meaning="",resource_type = self.resource_type , mode=self.mode,GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)
        # except:
        #     return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)

    def contribute_in_alriyadh_dictionary_add(self):
        try:
            word = request.form.get('word')
            meaning = request.form.get('meaning')
            resource_type = request.form.get('resource_type')
            word_type = request.form['type-select']
            example = request.form.get('example')

            self.add_data_to_csv("static/users_contributions.csv", word, example, meaning, resource_type, word_type, report=False, add=True)
            return render_template(f'setup_{self.setup}.html', word="",meaning="",resource_type = "",mode=self.mode,GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)

    def contribute_in_alriyadh_dictionary_report(self):
        try:
            word = request.form.get('word')
            meaning = request.form.get('meaning')
            resource_type = request.form.get('resource_type')
            word_type = request.form['type-select']
            example = request.form.get('example')

            self.add_data_to_csv("static/users_contributions.csv", word, example, meaning, resource_type, word_type, report=True, add=False)
            return render_template(f'setup_{self.setup}.html', word="",meaning="",resource_type = "",mode=self.mode,GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)


    def add_data_to_csv(self,file_path, word, example, meaning,resource_type, word_type, report=False, add=False):
        """
        Add data to a CSV file.

        Parameters:
        - file_path (str): The path to the CSV file.
        - word (str): The word to add.
        - example (str): An example of the word's usage.
        - meaning (str): The meaning of the word.
        - report (bool): Whether to report the word (default is False).
        - add (bool): Whether to add the word (default is False).
        """
        fieldnames = ["word", "example", "meaning", "resource_type", "word_type", "report", "add"]

        # Check if the file exists, and create it with header if not
        file_exists = False
        try:
            with open(file_path, 'r') as file:
                file_exists = True
        except FileNotFoundError:
            pass

        with open(file_path, 'a', newline='',encoding="UTF-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write header only if the file is newly created
            if not file_exists:
                writer.writeheader()

            # Write data
            writer.writerow({
                "word": word,
                "example": example,
                "meaning": meaning,
                "resource_type":resource_type,
                "word_type":word_type,
                "report": report,
                "add": add,

            })

    def get_list_of_meanings_dicts_for_word(self,word):
        url = "https://siwar.ksaa.gov.sa/api/alriyadh/search"
        query_param = word

        headers = {
            "apikey": API_KEY
        }

        params = {
            "query": query_param,
        }

        # Make the GET request with certificate verification
        response = requests.get(url, headers=headers, params=params, verify=False)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse and work with the response data (assuming it's in JSON format)
            data = response.json()

            list_of_meanings_dicts = []
            id = 0
            for i in range(len(data)):
                if len(data[i]['senses'])>0:
                    for dic in data[i]['senses']:
                        if dediac_ar(word) == dediac_ar(str(data[i]['nonDiacriticsLemma'])):
                            meaning_dict = {}
                            meaning_dict["id"] = id
                            meaning_dict["meaning"] = str(dic['definition']['textRepresentations'][0]['form'])
                            meaning_dict['nonDiacriticsLemma'] = str(data[i]['nonDiacriticsLemma'])
                            meaning_dict["word_with_diacr"] = str(data[i]['nonDiacriticsLemma'])
                            meaning_dict["html"] =  self.extract_window_around_word(text=meaning_dict['word_with_diacr'] + ": " + meaning_dict["meaning"],target_word=meaning_dict['word_with_diacr']+ ":",above_n_chars=55,window_size=7)

                            meaning_dict["pos"] = str(data[i]['pos'])
                            if len(meaning_dict["meaning"]) > 0:
                                list_of_meanings_dicts.append(meaning_dict)
                                id+=1

        return list_of_meanings_dicts

    def set_mode(self):
        try:
            mode = request.form.get('mode')
            self.mode = mode
            return render_template(f'setup_{self.setup}.html', examples=[], word="",meaning="", word_type= "Noun", mode=self.mode,resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)
        except:
            return render_template(f'setup_1.html',examples=[], word="", meaning="", mode="light", word_type="Noun", resource_type = "all",GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2=GROUP_EXAMPLES_BY_WORD_IN_DISPLAY_WAY_2)

    def move_tuple_to_beginning(self,lst, target_value):
        for i, item in enumerate(lst):
            if item[0] == target_value:
                # Swap the current tuple with the first tuple
                lst[0], lst[i] = lst[i], lst[0]
                break

    def get_model_and_tokenizer(self):
        if MODEL_NAME == "arabert_base":
            saved_model_path = "static/resources/arabert_base/bert-base-arabertv02"
            huggingface_hub_model_name = "aubmindlab/bert-base-arabertv02"

        if MODEL_NAME == "arabert_large":
            saved_model_path = "static/resources/arabert_large/bert-large-arabertv02"
            huggingface_hub_model_name = "aubmindlab/bert-large-arabertv02"

        if MODEL_NAME == "Arabic_KW_Mdel" or MODEL_NAME == "KW_Model_without_SW":
            saved_model_path = "static/resources/Arabic_KW_Mdel/Arabic-KW-Mdel"
            huggingface_hub_model_name = "medmediani/Arabic-KW-Mdel"

        if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
            tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
            model = AutoModel.from_pretrained(saved_model_path)
        else:

            tokenizer = AutoTokenizer.from_pretrained(huggingface_hub_model_name)
            model = AutoModel.from_pretrained(huggingface_hub_model_name)

            tokenizer.save_pretrained(saved_model_path)
            model.save_pretrained(saved_model_path)


        return model.to(device), tokenizer


    def get_all_inverted_indices(self):
        unique = set()
        inverted_indices = {}
        for _type in tqdm(SUPPORTED_EXAMPLES_TYPES,desc="get_all_inverted_indices()"):
            # Load the list from the file
            with open(f"static/resources/{MODEL_NAME}/{_type}/inverted_index.pkl", 'rb') as file:
                inverted_indices[_type] = pickle.load(file)
                # print(_type,len(inverted_indices[_type]))
                for key in inverted_indices[_type]:
                    unique.add(key)
        # print(f"unique:{len(unique)}")
        return inverted_indices

    def get_types_metadata(self):
        types_metadata = {}
        current_types_require_metadata = [_type for _type in EXAMPLES_TYPES_REQUIRE_METADATA if _type in SUPPORTED_EXAMPLES_TYPES]
        for _type in tqdm(current_types_require_metadata,desc="get_types_metadata()"):
            with open(f"static/resources/{MODEL_NAME}/{_type}/{_type}_index_to_meta_data_dict.pkl", 'rb') as file:
                types_metadata[_type] = pickle.load(file)
        return types_metadata

    def get_all_sentences(self):
        unique = set()

        sentences = {}
        for _type in tqdm(SUPPORTED_EXAMPLES_TYPES,desc="get_all_sentences()"):
            # Load the list from the file
            with open(f"static/resources/{MODEL_NAME}/{_type}/Processed_sentences.pkl", 'rb') as file:
                sentences[_type] = pickle.load(file)
                # print(_type,len(sentences[_type]))
                for sent in sentences[_type]:
                    unique.add(sent)
        # print(f"unique:{len(unique)}")
        return sentences

    def get_all_embeddings(self):
        embeddings = {}
        for _type in tqdm(SUPPORTED_EXAMPLES_TYPES,desc="get_all_embeddings()"):
            loaded_tensor_dict = torch.load(f"static/resources/{MODEL_NAME}/{_type}/embeddings.pt",map_location=device)
            embeddings_list = []
            # Iterate over the dictionary values (tensors) and add them to the list
            for tensor_list in loaded_tensor_dict.values():
                for tensor in tensor_list:
                    embeddings_list.append(tensor)
            embeddings[_type] = embeddings_list

        return embeddings

    def make_queries_embeddings(self, words, meaning, word_type):
        queries = [f"{word}:{meaning}" for word in words]
        input_ids = self.tokenizer(queries, return_tensors="pt", padding=True).to(device)
        # Get the model's output
        with torch.no_grad():
            output = self.model(**input_ids)

        # Get the embeddings from the output
        queries_embeddings = output.last_hidden_state
        # Get the Average over the 1st dim
        queries_embeddings = torch.mean(queries_embeddings, dim=1)

        if len(words) == 1:
            return [queries_embeddings]
        else:
            return torch.split(queries_embeddings, 1, dim=0)

    def remove_near_distances(self,distances, indices, threshold):
        new_distances = []
        new_indices = []

        for i in range(len(distances)):
            current_distance = distances[i]
            should_add = True

            # Check if the current distance is near any existing distance
            for j in range(len(new_distances)):
                existing_distance = new_distances[j]

                if abs(current_distance - existing_distance) < threshold:
                    should_add = False
                    break

            if should_add:
                new_distances.append(current_distance)
                new_indices.append(indices[i])

        return new_distances, new_indices

    def get_examples_with_faiss(self,word ,meaning, word_type, resource_type):

        if REMOVE_STOPWORDS_FROM_MEANING:
            meaning_filtered_words = [word for word in meaning.split() if dediac_ar(word) not in self.arabic_stopwords]
            # Join the filtered words back into a sentence
            meaning = ' '.join(meaning_filtered_words)

        if REMOVE_EXCLUDED_WORDS_FROM_MEANING:
            meaning_filtered_words = [word for word in meaning.split() if dediac_ar(word) not in Excluded_Words_From_Meanings]
            # Join the filtered words back into a sentence
            meaning = ' '.join(meaning_filtered_words)

        if REMOVE_WORD_FROM_MEANING:
            for word_in_meaning in meaning.split():
                if dediac_ar(word_in_meaning) == word:
                    meaning = meaning.replace(word_in_meaning+" ","")
                    meaning = meaning.replace(" "+word_in_meaning,"")



        print(f"Query: word:{word} ,meaning:{meaning}, word_type:{word_type}, resource_type:{resource_type}")
        self.write_to_logs(f"Query: word:{word} ,meaning:{meaning}, word_type:{word_type}, resource_type:{resource_type}")

        lemma_custom_checkbox = True
        examples_dict = {}
        current_resource_types = []
        if resource_type == "all":
            for _type in SUPPORTED_EXAMPLES_TYPES:
                current_resource_types.append(_type)
        else:
            current_resource_types.append(resource_type)

        for _type in current_resource_types:
            start_time = time.time()
            all_forms_of_word = self.get_all_forms_of_word(_type, lemma_custom_checkbox, word, word_type)
            print(all_forms_of_word)
            self.write_to_logs(all_forms_of_word)

            if len(all_forms_of_word) > 0:
                queries_embeddings = self.make_queries_embeddings(all_forms_of_word, meaning, word_type)
            else:
                continue
            current_resource_type_all_words_forms_examples = []
            for word_form, query_embedding in zip(all_forms_of_word,queries_embeddings):
                words_form_examples = []

                # Filter the embeddings to contain only the embeddings of the word (to search only with those embeddings)
                inverted_indicies = self.inverted_indices[_type][word_form]
                faiss_index_to_sentence_index_dict = {faiss_output_index:sentence_index for faiss_output_index,sentence_index in enumerate(inverted_indicies)}
                embeddings = []
                for index in inverted_indicies:
                    embeddings.append(self.embeddings[_type][index])
                # # Stack the tensors into a single tensor
                filtered_embeddings = torch.stack(embeddings)

                # Ensure that (filtered_embeddings) is a 2D array of shape (num_embeddings, embedding_dim)
                embedding_np = filtered_embeddings.cpu().numpy()

                # # Create a Faiss index
                # index = faiss.IndexFlatL2(embedding_np.shape[1])  # L2 (Euclidean) distance index
                #
                # # Add the embeddings to the Faiss index
                # index.add(embedding_np)
                #
                # # Define the number of nearest neighbors you want to retrieve
                # k = len(self.inverted_indices[_type][word_form])
                #
                # # Define a query embedding for which you want to find similar embeddings
                # query_embedding = np.array(query_embedding.cpu(), dtype=np.float32)  # Replace with your query embedding
                #
                # # Perform a search to find the nearest neighbors
                # distances, indices = index.search(query_embedding, k)

                # Create a Faiss index with Cosine similarity
                index = faiss.IndexFlatIP(embedding_np.shape[1])  # Inner product (Cosine similarity) index

                # Normalize the embeddings to unit length for Cosine similarity
                faiss.normalize_L2(embedding_np)

                # Add your embeddings to the Faiss index
                index.add(embedding_np)

                # Define the number of nearest neighbors you want to retrieve
                k = len(self.inverted_indices[_type][word_form])

                # Define a query embedding for which you want to find similar embeddings
                query_embedding = np.array(query_embedding.cpu(), dtype=np.float32)  # Replace with your query embedding
                query_embedding /= np.linalg.norm(query_embedding)  # Normalize the query embedding to unit length

                # Perform a search to find the nearest neighbors
                distances, indices = index.search(query_embedding, k)

                # Set a cosine similarity threshold between 0 and 1
                cosine_similarity_threshold = EXAMPLE_THRESHOLD  # Adjust the threshold as needed

                # Filter the results based on the cosine similarity threshold
                indices = [i for i, sim in zip(indices[0], distances[0]) if sim >= cosine_similarity_threshold]
                distances = [sim for i, sim in zip(indices, distances[0]) if sim >= cosine_similarity_threshold]


                distances, indices = self.remove_near_distances(indices = indices, distances=distances, threshold = REMOVE_SIMILAR_EXAMPLES_THRESHOLD)

                faiss_indices = indices
                faiss_distances = distances
                orignal_sentences = []
                distances = []

                for faiss_index ,faiss_distance in zip(faiss_indices,faiss_distances):
                    sentence_index = faiss_index_to_sentence_index_dict[faiss_index]
                    if REMOVE_SENTENCE_IF_BECOME_SHORT_AFTER_REMOVING_STOPWORDS:
                        list_of_sentence_words = self.sentences[_type][sentence_index].split()
                        intial_lenght = len(list_of_sentence_words)
                        list_of_sentence_words = [sentence_word for sentence_word in list_of_sentence_words if sentence_word not in self.arabic_stopwords]
                        if len(list_of_sentence_words) != intial_lenght and len(list_of_sentence_words) < REMOVE_SENTENCE_IF_BECOME_SHORT_AFTER_REMOVING_STOPWORDS_N:
                            continue
                    words_form_examples.append(self.generate_html_sentence(sentence=self.sentences[_type][sentence_index], word_to_highlight=word_form,example_type=_type,sentence_index=sentence_index,distance=faiss_distance))
                    orignal_sentences.append(self.sentences[_type][sentence_index])
                    distances.append(faiss_distance)
                    self.write_to_logs(f"word:{word_form}, example:{self.sentences[_type][sentence_index]}")
                current_resource_type_all_words_forms_examples.append((word_form, words_form_examples, distances,orignal_sentences))
            print("--- %s seconds ---" % (time.time() - start_time))

            examples_dict[_type] = current_resource_type_all_words_forms_examples

        # Create a defaultdict with default values as empty lists
        words_examples = defaultdict(list) # for DISPLAY_WAY == 1
        flattend_list_of_examples_dicts = [] # for DISPLAY_WAY == 2
        for examples_type in examples_dict.keys():
            for word_examples_distances_orignal_sentences_tuple in examples_dict[examples_type]:
                word, examples, distances,orignal_sentences = word_examples_distances_orignal_sentences_tuple
                for example,distance,orignal_sentence in zip(examples[:MAX_NUM_OF_EXAMPLES_PER_WORD], distances[:MAX_NUM_OF_EXAMPLES_PER_WORD], orignal_sentences[:MAX_NUM_OF_EXAMPLES_PER_WORD]):
                    words_examples[word].append({"example":example,"distance":distance,"type":examples_type,'orignal_sentence':orignal_sentence})
                    flattend_list_of_examples_dicts.append({"word": word,"example":[example],"distance":distance,"type":examples_type,'orignal_sentence':orignal_sentence})

        examples_list = []

        if DISPLAY_WAY == 1:
            # Sorting the examples based on the distance from the query (not by the type) in descending order
            for word in words_examples.keys():
                words_examples[word] = sorted(words_examples[word], key=lambda x: x['distance'], reverse=True)
            # Sorting the examples based on the distance from the query (not by the type)

            # Sort python_dict based on the average of "distance" values for each key
            sorted_dict = {}
            for word, values in words_examples.items():
                avg_distance = sum(d['distance'] for d in values) / len(values)
                sorted_dict[word] = avg_distance
            # Sort the dictionary based on average distance in descending order
            sorted_dict = dict(sorted(sorted_dict.items(), key=lambda item: item[1], reverse=True))
            words_examples = {key: words_examples[key] for key in sorted_dict}
            # Sort python_dict based on the average of "distance" values for each key

            # Form the html needed output list of (word,examples),(word,examples),(word,examples),...

            for word in words_examples.keys():
                examples = []
                # print(f"Word:{word}")
                self.write_to_logs(f"Word:{word}")
                for dic in words_examples[word]:
                    examples.append(dic['example'])
                    # print(f"\t-sentence:{dic['orignal_sentence']}\n\t- distance:{dic['distance']}\n\n")
                    self.write_to_logs(f"\t-sentence:{dic['orignal_sentence']}\n\t- distance:{dic['distance']}\n\n")
                # print(f"-------------------")
                self.write_to_logs(f"-------------------")
                examples_list.append((word,examples))
            # Form the html needed output list of (word,examples),(word,examples),(word,examples),...

        elif DISPLAY_WAY == 2:
            # Sort by score absolutely
            flattend_list_of_examples_dicts = sorted(flattend_list_of_examples_dicts, key=lambda x: x["distance"],reverse=True)

            # Form the html needed output list of (word,examples),(word,examples),(word,examples),...


            # Initialize variables to store the current word and examples
            current_word = None
            current_examples = []

            if len(flattend_list_of_examples_dicts) > 0:
                for item in flattend_list_of_examples_dicts:
                    word = item["word"]
                    examples = item["example"]

                    if current_word is None:
                        # For the first item, simply set the current_word and current_examples
                        current_word = word
                        current_examples = examples
                    elif current_word == word:
                        # If the current word is the same as the previous one, add their example lists
                        current_examples.extend(examples)
                    else:
                        # If the current word is different, add the tuple (current_word, current_examples) to the result list
                        examples_list.append((current_word, current_examples))
                        # Reset current_word and current_examples for the new word
                        current_word = word
                        current_examples = examples

                # Add the last tuple to the result list
                examples_list.append((current_word, current_examples))

            # Form the html needed output list of (word,examples),(word,examples),(word,examples),...

        return examples_list

    def extract_window_around_word(self, text, target_word,above_n_chars, window_size=TEXT_WINDOW_SIZE):
        if len(text) > above_n_chars:
            # Split the text into words
            words = text.split()

            # Find the index of the target word
            try:
                word_index = words.index(target_word)
            except:
                # Target word not found in the text
                return text

            # Calculate the start and end indices for the window
            start_index = max(0, word_index - window_size)
            end_index = min(len(words), word_index + window_size + 1)

            # Extract the window around the target word
            window = words[start_index:end_index]

            # Join the words in the window to form the final text
            result_text = ' '.join(window)

            if start_index > 0:
                result_text = "..." + result_text

            if end_index < len(words):
                result_text = result_text +"..."

            return result_text
        else:
            return text

    def get_all_forms_of_word(self, _type, lemma_custom_checkbox, word, word_type):
        all_forms_of_word = []
        all_forms_of_word.append(word)
        if lemma_custom_checkbox:
            forms_of_word = self.get_all_forms_of_arabic_word(word, word_type)
            for word_form in forms_of_word:
                all_forms_of_word.append(word_form)
        # Add non-dediac words also
        all_forms_of_word = all_forms_of_word + [dediac_ar(word_form) for word_form in all_forms_of_word]
        all_forms_of_word = list(set(all_forms_of_word))
        all_forms_of_word = [word_form for word_form in all_forms_of_word if
                             len(self.inverted_indices[_type][word_form]) > 0]
        # if word in all_forms_of_word:
        #     all_forms_of_word.remove(word)
        #     all_forms_of_word.insert(0, word)
        return all_forms_of_word

    def generate_html_sentence(self, sentence, word_to_highlight,example_type,sentence_index,distance):
        orignal_sentence = sentence
        sentence = self.extract_window_around_word(sentence,word_to_highlight,above_n_chars=150)
        if example_type == "quraan":
            meta_data = self.types_metadata['quraan'][sentence_index]
            sentence = "﴿ "+sentence+" ﴾" + f" [{meta_data['sura']}:{meta_data['aya']}]"
            orignal_sentence = "﴿ "+orignal_sentence+" ﴾" + f" [{meta_data['sura']}:{meta_data['aya']}]"
            # sentence = "﴾"+sentence+"﴿"+ "[" + "سورة: " + meta_data['sura'] + " آية: " + str(meta_data['aya'])

        if example_type == "hadith":
            meta_data = self.types_metadata['hadith'][sentence_index]
            sentence = "("+sentence+")"+ f" [{meta_data}]"
            orignal_sentence = "("+orignal_sentence+")"+ f" [{meta_data}]"

        pattern_arabic_with_diacritics = re.compile(r'[\u0600-\u06FF]+')
        matches_arabic = pattern_arabic_with_diacritics.finditer(sentence)

        # Initialize the words_positions_in_sentences as a defaultdict of lists
        words_positions_in_sentences = defaultdict(list)

        for match in matches_arabic:
            start_pos = match.start()
            end_pos = match.end()
            match_text = match.group(0)
            words_positions_in_sentences[match_text].append((start_pos, end_pos))

        red_words = words_positions_in_sentences[word_to_highlight]

        html_output = f'<span title="نسبة التشابه: %{str(distance * 100)[:5]}\n{orignal_sentence}">'
        # html_output = f'<span title="{orignal_sentence}">'


        current_pos = 0
        for start, end in red_words:
            html_output += sentence[current_pos:start]  # Append non-red part
            html_output += f'<span class="word_to_highlight">{sentence[start:end]}</span>'  # Append red part with underline
            current_pos = end

        html_output += sentence[current_pos:]  # Append the remaining non-red part

        if example_type == "quraan":
            html_output += '<span title="موقع تنزيل" style="font-size: 14px;margin-right: 6px;">مصدر النص القرآني : <a href="https://tanzil.net/" style="color: Black;">موقع تنزيل</a></span>'

        html_output += '</span>'
        return html_output

    def get_all_forms_of_arabic_word(self,word, word_type):
        all_forms = []

        """
        asp - Aspect
            c - Command
            i - Imperfective
            p - Perfective
            na - Not applicable
        
        gen - Gender
            f - Feminine
            m - Masculine
            na - Not applicable
        
        mod - Mood
            i - Indicative
            j - Jussive
            s - Subjunctive
            na - Not applicable
            u - Undefined
        
        num - Number
            s - Singular
            d - Dual
            p - Plural
            na - Not applicable
            u - Undefined
        
        pos - Part-of-speech
            noun - Noun
            adj - Adjective
            pron - Pronoun
            verb - Verb
            part - Particle
            prep - Preposition
            abbrev - Abbreviation
            punc - Punctuation
            conj - Conjunction
        """
        pos = {
            "Noun": ['noun',  'adj', 'pron'],
            "Verb": ['verb'],
            "Preposition": ['prep']
        }
        features = {
            'asp': ['p', 'i', 'c', 'na'],
            'gen': ['f', 'm', 'na', 'u'],
            'mod': ['i', 'j', 's', 'na', 'u'],
            'num': ['s', 'd', 'p', 'na', 'u'],
            'pos': pos[word_type],
        }

        # Create combinations of feature values
        feature_combinations = product(*(features[feature] for feature in features))

        for feature_values in feature_combinations:
            feature_dict = dict(zip(features, feature_values))
            analyses = self.reinflector.reinflect(word, feature_dict)
            unique_diacritizations = set(a['diac'] for a in analyses)
            all_forms.extend(unique_diacritizations)
        return all_forms


    def write_to_logs(self,data):
        # Open the file in append mode ('a' flag) to create if it doesn't exist and append if it does
        with open('logs.txt', 'a',encoding='UTF-8') as file:
            # Write the input data to the file with a newline character
            file.write(str(data) + '\n')

    def load_arabic_stopwords(self):
        with open("static/resources/arabic_stopwords.txt", 'r',encoding="UTF-8") as file:
            arabic_stopwords = [line.strip() for line in file]

        return arabic_stopwords

if __name__ == '__main__':
    my_app = MustashhedApp(__name__)
    my_app.app.run(debug=False,host="0.0.0.0")