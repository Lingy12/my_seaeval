#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Created Date: Tuesday, July 25th 2023, 10:47:48 am
# Author: Bin Wang
# -----
# Copyright (c) Bin Wang @ bwang28c@gmail.com
# 
# -----
# HISTORY:
# Date&Time 			By	Comments
# ----------			---	----------------------------------------------------------
###


# add parent directory to sys.path
import sys
sys.path.append('.')

import random

import re
import logging

from collections import Counter

from nltk.tokenize import sent_tokenize

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from rouge_score import rouge_scorer
from bert_score import BERTScorer

from config.config import (
    DATASET_TYPE,
)

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

class Metric(object):
    
    def __init__(self, dataset_name: str=""):

        self.dataset_name = dataset_name
        self.dataset_type = DATASET_TYPE[dataset_name]
        self._load_metric()


    def _load_metric(self):

        if self.dataset_type == 'eng_summarization':
            self._load_metric_eng_summarization()

        elif self.dataset_type == 'eng_multi_choice_context':
            self._load_metric_eng_multi_choice_context()

        elif self.dataset_type == 'eng_classification':
            self._load_metric_eng_classification()

        elif self.dataset_type == 'eng_classification_multi_choice':
            self._load_metric_eng_classification_multi_choice()

        elif self.dataset_type == 'eng_multi_choice_no_context':
            self._load_metric_eng_multi_choice_no_context()

        elif self.dataset_type == 'chi_multi_choice_no_context':
            self._load_metric_chi_multi_choice_no_context()

        elif self.dataset_type == 'ind_classification_multi_choice':
            self._load_metric_ind_classification_multi_choice()

        elif self.dataset_type == 'to_eng_translation':
            self._load_metric_to_eng_translation()
        
        elif self.dataset_type == 'chi_ocnli_multi_choice':
            self._load_metric_chi_multi_choice_no_context()

        elif self.dataset_type == 'chi_c3_multi_choice':
            self._load_metric_chi_multi_choice_no_context()

        elif self.dataset_type == 'cross_mmlu':
            self._load_metric_cross_mmlu()

        elif self.dataset_type == 'local_eval':
            pass

        elif self.dataset_type == 'cross_logiqa':
            pass

        else:
            raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))
        

    def compute(self, data, predictions):
            
        if self.dataset_type == 'eng_summarization':
            return self._compute_eng_summarization(data, predictions)
        
        elif self.dataset_type == 'eng_multi_choice_context':
            return self._compute_eng_multi_choice_context(data, predictions)
        
        elif self.dataset_type == 'eng_classification':
            return self._compute_eng_classification(data, predictions)
        
        elif self.dataset_type == 'eng_classification_multi_choice':
            return self._compute_eng_classification_multi_choice(data, predictions)
        
        elif self.dataset_type == 'eng_multi_choice_no_context':
            return self._compute_eng_multi_choice_context(data, predictions)
        
        elif self.dataset_type == 'chi_multi_choice_no_context':
            return self._compute_chi_multi_choice_no_context(data, predictions)
        
        elif self.dataset_type == 'ind_classification_multi_choice':
            return self._compute_ind_classification_multi_choice(data, predictions)

        elif self.dataset_type == 'to_eng_translation':
            return self._compute_to_eng_translation(data, predictions)
        
        elif self.dataset_type == 'chi_ocnli_multi_choice':
            return self._compute_chi_multi_choice_no_context(data, predictions)

        elif self.dataset_type == 'chi_c3_multi_choice':
            return self._compute_chi_multi_choice_no_context(data, predictions)
        
        elif self.dataset_type == 'cross_mmlu':
            return self._compute_cross_mmlu(data, predictions)

        elif self.dataset_type == 'local_eval':
            return self._compute_local_eval(data, predictions)
        
        elif self.dataset_type == 'cross_logiqa':
            return self._compute_cross_mmlu(data, predictions)

        else:
            raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))


    def _load_metric_eng_summarization(self):

        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, split_summaries=True)

    def _load_metric_eng_multi_choice_context(self):

        self.bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

    def _load_metric_eng_classification(self):

        self.bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

    def _load_metric_eng_classification_multi_choice(self):

        self.bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

    def _load_metric_eng_multi_choice_no_context(self):

        self.bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True)

    def _load_metric_chi_multi_choice_no_context(self):

        logger.info('Matching-based metrics are prepared for {} dataset.'.format(self.dataset_type))

    def _load_metric_ind_classification_multi_choice(self):

        logger.info('Matching-based metrics are prepared for Indonesian classification multi-choice dataset.')

    def _load_metric_to_eng_translation(self):

        self.bleu_smooth_function = SmoothingFunction()
        self.sentence_bleu = sentence_bleu

    def _load_metric_cross_mmlu(self):

        logger.info('Matching-based metrics are prepared for cross-lingual MMLU dataset.')







    def _compute_eng_summarization(self, data, predictions):

        rouge1 = 0
        rouge2 = 0
        rougeL = 0

        for i in range(len(data)):
            scores = self.rouge_scorer.score(data[i]['output'], predictions[i])
            rouge1 += scores['rouge1'].fmeasure
            rouge2 += scores['rouge2'].fmeasure
            rougeL += scores['rougeL'].fmeasure

            '''
            logger.info("\n[Sample Index]:\n{}:".format(data[i]['id']))
            logger.info("\n[Input]:\n{}".format(data[i]['input']))
            logger.info("\n[Groundtruth]:\n{}".format(data[i]['output']))
            logger.info("\n[Model Prediction]:\n{}".format(predictions[i]))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''


        rouge1 /= len(data)
        rouge2 /= len(data)
        rougeL /= len(data)

        avg_rouge = (rouge1 + rouge2 + rougeL) / 3

        results = {
            'rouge1'   : rouge1,
            'rouge2'   : rouge2,
            'rougeL'   : rougeL,
            'avg_rouge': avg_rouge,
        }

        return results
    
    def _compute_eng_multi_choice_context(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        matching = 0
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            # expand choices candidates
            expanded_choices_mapping = {
                choices[0].lower(): choices[0],
                choices[1].lower(): choices[1],
                "A".lower(): choices[0],
                "B".lower(): choices[1],
                "(A)".lower(): choices[0],
                "(B)".lower(): choices[1],
                "(A".lower(): choices[0],
                "(B".lower(): choices[1],
                "A)".lower(): choices[0],
                "B)".lower(): choices[1],
            }
            if len(choices) >= 3:
                expanded_choices_mapping[choices[2].lower()] = choices[2]
                expanded_choices_mapping["C".lower()] = choices[2]
                expanded_choices_mapping["(C)".lower()] = choices[2]
                expanded_choices_mapping["(C".lower()] = choices[2]
                expanded_choices_mapping["C)".lower()] = choices[2]
                
            if len(choices) >= 4:
                expanded_choices_mapping[choices[3].lower()] = choices[3]
                expanded_choices_mapping["D".lower()] = choices[3]
                expanded_choices_mapping["(D)".lower()] = choices[3]
                expanded_choices_mapping["(D".lower()] = choices[3]
                expanded_choices_mapping["D)".lower()] = choices[3]

            if len(choices) >= 5:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))
            
            choices_list = list(expanded_choices_mapping.keys())
            
            prediction_sent = sent_tokenize(prediction.replace('\n', '. '))

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.lower().strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
                if not sent: continue

                scores = self.bert_scorer.score([sent]*len(choices_list), choices_list)[2]
                scores = scores.tolist()

                if max(scores) < 0.2: continue

                acceptable_threshold = max(scores) - 0.02
                chosen_answers = [choices_list[i] for i, score in enumerate(scores) if score >= acceptable_threshold]
                all_chosen_answers.extend(chosen_answers)

          
            if len(all_chosen_answers) == 0:
                answer = "No Answer."

            else:
                chosen_labels = [expanded_choices_mapping[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]
            
            if answer == "No Answer.":
                answer = random.choice(choices)

            if answer == output:
                matching += 1
            '''
            logger.info("\n[Model Prediction]:\n{}".format(prediction))
            logger.info("\n[Choices]:\n{}".format(choices))
            logger.info("\n[All chosen answers]:\n{}".format(all_chosen_answers))
            logger.info("\n[Model Answer]:\n{}".format(answer))
            logger.info("\n[Groundtruth]:\n{}".format(output))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''


        accuracy = matching / len(data)

        return {'Accuracy': accuracy}
            
            
    def _compute_eng_classification(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        matching = 0
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            choices_list = list(choices.keys())

            prediction_sent = sent_tokenize(prediction.replace('\n', '. '))

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.lower().strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
                if not sent: continue

                scores = self.bert_scorer.score([sent]*len(choices_list), choices_list)[2]
                scores = scores.tolist()

                if max(scores) < 0.2: continue

                acceptable_threshold = max(scores) - 0.02
                chosen_answers = [choices_list[i] for i, score in enumerate(scores) if score >= acceptable_threshold]
                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) == 0:
                answer = "No Answer."
            
            else:
                chosen_labels = [choices[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]

            if answer == 'No Answer.':
                answer = random.choice(list(choices.values()))

            if answer == output:
                matching += 1
            
            '''
            logger.info("\n[Model Prediction]:\n{}".format(prediction))
            logger.info("\n[Choices]:\n{}".format(choices))
            logger.info("\n[All chosen answers]:\n{}".format(all_chosen_answers))
            logger.info("\n[Model Answer]:\n{}".format(answer))
            logger.info("\n[Groundtruth]:\n{}".format(output))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''


        accuracy = matching / len(data)

        return {'Accuracy': accuracy}


    def _compute_eng_classification_multi_choice(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        matching = 0
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            # expand choices candidates
            expanded_choices_mapping = {
                choices[0].lower(): choices[0],
                choices[1].lower(): choices[1],
                "A".lower(): choices[0],
                "B".lower(): choices[1],
                "(A)".lower(): choices[0],
                "(B)".lower(): choices[1],
                "(A".lower(): choices[0],
                "(B".lower(): choices[1],
                "A)".lower(): choices[0],
                "B)".lower(): choices[1],
            }
            if len(choices) >= 3:
                expanded_choices_mapping[choices[2].lower()] = choices[2]
                expanded_choices_mapping["C".lower()] = choices[2]
                expanded_choices_mapping["(C)".lower()] = choices[2]
                expanded_choices_mapping["(C".lower()] = choices[2]
                expanded_choices_mapping["C)".lower()] = choices[2]
                
            if len(choices) >= 4:
                expanded_choices_mapping[choices[3].lower()] = choices[3]
                expanded_choices_mapping["D".lower()] = choices[3]
                expanded_choices_mapping["(D)".lower()] = choices[3]
                expanded_choices_mapping["(D".lower()] = choices[3]
                expanded_choices_mapping["D)".lower()] = choices[3]

            if len(choices) >= 5:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))
            
            choices_list = list(expanded_choices_mapping.keys())
            
            prediction_sent = sent_tokenize(prediction.replace('\n', '. '))

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.lower().strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
                if not sent: continue

                scores = self.bert_scorer.score([sent]*len(choices_list), choices_list)[2]
                scores = scores.tolist()

                if max(scores) < 0.2: continue

                acceptable_threshold = max(scores) - 0.02
                chosen_answers = [choices_list[i] for i, score in enumerate(scores) if score >= acceptable_threshold]
                all_chosen_answers.extend(chosen_answers)

          
            if len(all_chosen_answers) == 0:
                answer = "No Answer."

            else:
                chosen_labels = [expanded_choices_mapping[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]

            if answer == 'No Answer.':
                answer = random.choice(choices)

            if answer == output:
                matching += 1

            '''
            logger.info("\n[Model Prediction]:\n{}".format(prediction))
            logger.info("\n[Choices]:\n{}".format(choices))
            logger.info("\n[All chosen answers]:\n{}".format(all_chosen_answers))
            logger.info("\n[Model Answer]:\n{}".format(answer))
            logger.info("\n[Groundtruth]:\n{}".format(output))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''

        accuracy = matching / len(data)

        return {'Accuracy': accuracy}



    def _compute_chi_multi_choice_no_context(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        matching = 0
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            # expand choices candidates
            expanded_choices_mapping = {
                choices[0]: choices[0],
                "A "       : choices[0],
                " A"       : choices[0],
                "(A)"     : choices[0],
                "(A"      : choices[0],
                "A)"      : choices[0],
                choices[1]: choices[1],
                "B "       : choices[1],
                " B"       : choices[1],
                "(B)"     : choices[1],
                "(B"      : choices[1],
                "B)"      : choices[1],
            }

            if len(choices) >= 3:
                expanded_choices_mapping[choices[2]] = choices[2]
                expanded_choices_mapping["C "] = choices[2]
                expanded_choices_mapping[" C"] = choices[2]
                expanded_choices_mapping["(C)"] = choices[2]
                expanded_choices_mapping["(C"] = choices[2]
                expanded_choices_mapping["C)"] = choices[2]

            if len(choices) >= 4:
                expanded_choices_mapping[choices[3]] = choices[3]
                expanded_choices_mapping["D "] = choices[3]
                expanded_choices_mapping[" D"] = choices[3]
                expanded_choices_mapping["(D)"] = choices[3]
                expanded_choices_mapping["(D"] = choices[3]
                expanded_choices_mapping["D)"] = choices[3]

            if len(choices) >= 5:
                expanded_choices_mapping[choices[4]] = choices[4]
                expanded_choices_mapping["E "] = choices[4]
                expanded_choices_mapping[" E"] = choices[4]
                expanded_choices_mapping["(E)"] = choices[4]
                expanded_choices_mapping["(E"] = choices[4]
                expanded_choices_mapping["E)"] = choices[4]

            if len(choices) >= 6:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

            choices_list = list(expanded_choices_mapping.keys())

            # prediction_sent = sent_tokenize(prediction.replace('\n', '。')) # nltk does not support chinese

            prediction_trun = prediction.replace('\n', '。')
            prediction_sent = re.split('(。|！|\!|\.|？|\?)', prediction_trun)

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.strip("(。$,|！|\!|\.|？|\?)，；：？！…—·《》“”‘’{}[]（）()、|\\/\n\t\r\v\f ")
                if len(sent) == 1: sent = sent + " "
                if not sent: continue

                chosen_answers = [item for item in choices_list if sent.startswith(item) or sent.endswith(item)]
                if len(chosen_answers) > 1: chosen_answers = [chosen_answers[0]]

                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) == 0:
                answer = "No Answer."

            else:
                chosen_labels = [expanded_choices_mapping[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]

            if answer == 'No Answer.':
                answer = random.choice(choices)
                
            if answer == output:
                matching += 1

            '''
            logger.info("\n[Model Prediction]:\n{}".format(prediction))
            logger.info("\n[Choices]:\n{}".format(choices))
            logger.info("\n[All chosen answers]:\n{}".format(all_chosen_answers))
            logger.info("\n[Model Answer]:\n{}".format(answer))
            logger.info("\n[Groundtruth]:\n{}".format(output))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''


        accuracy = matching / len(data)

        return {'Accuracy': accuracy}
    

    def _compute_ind_classification_multi_choice(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        matching = 0
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            # expand choices candidates
            expanded_choices_mapping = {
                choices[0]: choices[0],
                "A "       : choices[0],
                " A"       : choices[0],
                "(A)"     : choices[0],
                "(A"      : choices[0],
                "A)"      : choices[0],
                choices[1]: choices[1],
                "B "       : choices[1],
                " B"       : choices[1],
                "(B)"     : choices[1],
                "(B"      : choices[1],
                "B)"      : choices[1],
            }
            if len(choices) >= 3:
                expanded_choices_mapping[choices[2]] = choices[2]
                expanded_choices_mapping["C "] = choices[2]
                expanded_choices_mapping[" C"] = choices[2]
                expanded_choices_mapping["(C)"] = choices[2]
                expanded_choices_mapping["(C"] = choices[2]
                expanded_choices_mapping["C)"] = choices[2]
            
            if len(choices) >= 4:
                expanded_choices_mapping[choices[3]] = choices[3]
                expanded_choices_mapping["D "] = choices[3]
                expanded_choices_mapping[" D"] = choices[3]
                expanded_choices_mapping["(D)"] = choices[3]
                expanded_choices_mapping["(D"] = choices[3]
                expanded_choices_mapping["D)"] = choices[3]

            if len(choices) >= 5:
                expanded_choices_mapping[choices[4]] = choices[4]
                expanded_choices_mapping["E "] = choices[4]
                expanded_choices_mapping[" E"] = choices[4]
                expanded_choices_mapping["(E)"] = choices[4]
                expanded_choices_mapping["(E"] = choices[4]
                expanded_choices_mapping["E)"] = choices[4]
            
            if len(choices) >= 6:
                expanded_choices_mapping[choices[5]] = choices[5]
                expanded_choices_mapping["F "] = choices[5]
                expanded_choices_mapping[" F"] = choices[5]
                expanded_choices_mapping["(F)"] = choices[5]
                expanded_choices_mapping["(F"] = choices[5]
                expanded_choices_mapping["F)"] = choices[5]
            
            if len(choices) >= 7:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))
            
            choices_list = list(expanded_choices_mapping.keys())

            prediction_sent = sent_tokenize(prediction.replace('\n', '. '))

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.strip(",._\"\'-+=!?()&^%$#@:\\|\{\}[]<>/`\n\t\r\v\f ")
                if len(sent) == 1: sent = sent + " "
                if not sent: continue

                chosen_answers = [item for item in choices_list if sent.startswith(item) or sent.endswith(item)]
                if len(chosen_answers) > 1: chosen_answers = [chosen_answers[0]]

                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) == 0:
                answer = "No Answer."

            else:
                chosen_labels = [expanded_choices_mapping[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]

            if answer == 'No Answer.':
                answer = random.choice(choices)
                
            if answer == output:
                matching += 1

            '''
            logger.info("\n[Model Prediction]:\n{}".format(prediction))
            logger.info("\n[Choices]:\n{}".format(choices))
            logger.info("\n[All chosen answers]:\n{}".format(all_chosen_answers))
            logger.info("\n[Model Answer]:\n{}".format(answer))
            logger.info("\n[Groundtruth]:\n{}".format(output))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''

        accuracy = matching / len(data)

        return {'Accuracy': accuracy}


    def _compute_to_eng_translation(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]

        bleu_sentence_scores = []
        for sentence, prediction in zip(all_outputs, predictions):
            bleu_sentence_scores.append(self.sentence_bleu(hypothesis=prediction, references=[sentence], smoothing_function=self.bleu_smooth_function.method1))

            '''
            logger.info("\n[Groundtruth]:\n{}".format(sentence))
            logger.info("\n[Model Prediction]:\n{}".format(prediction))
            logger.info("\n\n\n\n\n\n\n\n\n\n")
            '''


        bleu_score = sum(bleu_sentence_scores) / len(bleu_sentence_scores)

        return {'BLEU Score': bleu_score}
    

    def _compute_cross_mmlu(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        all_predictioned_mapped = []

        matching = 0
        match_one_by_one = []
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            # expand choices candidates
            expanded_choices_mapping = {
                choices[0]: choices[0],
                "A "       : choices[0],
                " A"       : choices[0],
                "(A)"     : choices[0],
                "(A"      : choices[0],
                "A)"      : choices[0],
                choices[1]: choices[1],
                "B "       : choices[1],
                " B"       : choices[1],
                "(B)"     : choices[1],
                "(B"      : choices[1],
                "B)"      : choices[1],
            }

            if len(choices) >= 3:
                expanded_choices_mapping[choices[2]] = choices[2]
                expanded_choices_mapping["C "] = choices[2]
                expanded_choices_mapping[" C"] = choices[2]
                expanded_choices_mapping["(C)"] = choices[2]
                expanded_choices_mapping["(C"] = choices[2]
                expanded_choices_mapping["C)"] = choices[2]

            if len(choices) >= 4:
                expanded_choices_mapping[choices[3]] = choices[3]
                expanded_choices_mapping["D "] = choices[3]
                expanded_choices_mapping[" D"] = choices[3]
                expanded_choices_mapping["(D)"] = choices[3]
                expanded_choices_mapping["(D"] = choices[3]
                expanded_choices_mapping["D)"] = choices[3]

            if len(choices) >= 5:
                expanded_choices_mapping[choices[4]] = choices[4]
                expanded_choices_mapping["E "] = choices[4]
                expanded_choices_mapping[" E"] = choices[4]
                expanded_choices_mapping["(E)"] = choices[4]
                expanded_choices_mapping["(E"] = choices[4]
                expanded_choices_mapping["E)"] = choices[4]

            if len(choices) >= 6:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

            choices_list = list(expanded_choices_mapping.keys())

            prediction_trun = prediction.replace('\n', '。')
            prediction_sent = re.split('(。|！|\!|\.|？|\?)', prediction_trun)

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.strip("(。$,|！|\!|\.|？|\?)，；：？！…—·《》“”‘’{}[]（）()、|\\/\n\t\r\v\f ")
                if len(sent) == 1: sent = sent + " "
                if not sent: continue

                chosen_answers = [item for item in choices_list if sent.startswith(item) or sent.endswith(item)]
                if len(chosen_answers) > 1: chosen_answers = [chosen_answers[0]]

                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) == 0:
                answer = "No Answer."

            else:
                chosen_labels = [expanded_choices_mapping[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]

            if answer not in choices:
                answer = random.choice(choices)

            if answer == output:
                matching += 1
                match_one_by_one.append(1)
            else:
                match_one_by_one.append(0)
                

            all_predictioned_mapped.append(answer)
            
        accuracy = matching / len(data)

        eng_acc = match_one_by_one[::3]
        eng_acc = sum(eng_acc) / len(eng_acc)

        zho_acc = match_one_by_one[1::3]
        zho_acc = sum(zho_acc) / len(zho_acc)

        ind_acc = match_one_by_one[2::3]
        ind_acc = sum(ind_acc) / len(ind_acc)

        # Consistency score
        consistency_match = 0
        for i in range(0, len(all_predictioned_mapped), 3):
            
            #try:
            #    print(consistency_match)
            #    print(all_predictioned_mapped[i])
            #    print(all_predictioned_mapped[i+1])
            #    print(all_predictioned_mapped[i+2])
            #except:
            #    import pdb; pdb.set_trace()

            if all_predictioned_mapped[i][0:4] == all_predictioned_mapped[i+1][0:4] == all_predictioned_mapped[i+2][0:4]:
                consistency_match += 1

            
        consistency_score = consistency_match / (len(all_predictioned_mapped)/3)

        # Harmonic mean of accuracy and consistency score
        ac3 = 2 * consistency_score * accuracy / (consistency_score + accuracy)
            
        return {'Accuracy': accuracy, 'Consistency Score': consistency_score, 'AC3': ac3, 'English Accuracy': eng_acc, 'Chinese Accuracy': zho_acc, 'Indonesian Accuracy': ind_acc}
                


    def _compute_local_eval(self, data, predictions):

        all_outputs = [sample['output'] for sample in data]
        all_choices = [sample['choices'] for sample in data]

        all_predictioned_mapped = []

        matching = 0
        match_one_by_one = []
        for prediction, choices, output in zip(predictions, all_choices, all_outputs):

            if prediction == "": prediction = "None"

            # expand choices candidates
            expanded_choices_mapping = {
                choices[0]: choices[0],
                "A "       : choices[0],
                " A"       : choices[0],
                "(A)"     : choices[0],
                "(A"      : choices[0],
                "A)"      : choices[0],
                choices[1]: choices[1],
                "B "       : choices[1],
                " B"       : choices[1],
                "(B)"     : choices[1],
                "(B"      : choices[1],
                "B)"      : choices[1],
            }

            if len(choices) >= 3:
                expanded_choices_mapping[choices[2]] = choices[2]
                expanded_choices_mapping["C "] = choices[2]
                expanded_choices_mapping[" C"] = choices[2]
                expanded_choices_mapping["(C)"] = choices[2]
                expanded_choices_mapping["(C"] = choices[2]
                expanded_choices_mapping["C)"] = choices[2]

            if len(choices) >= 4:
                expanded_choices_mapping[choices[3]] = choices[3]
                expanded_choices_mapping["D "] = choices[3]
                expanded_choices_mapping[" D"] = choices[3]
                expanded_choices_mapping["(D)"] = choices[3]
                expanded_choices_mapping["(D"] = choices[3]
                expanded_choices_mapping["D)"] = choices[3]

            if len(choices) >= 5:
                expanded_choices_mapping[choices[4]] = choices[4]
                expanded_choices_mapping["E "] = choices[4]
                expanded_choices_mapping[" E"] = choices[4]
                expanded_choices_mapping["(E)"] = choices[4]
                expanded_choices_mapping["(E"] = choices[4]
                expanded_choices_mapping["E)"] = choices[4]

            if len(choices) >= 6:
                raise NotImplementedError("Dataset type {} not implemented yet".format(self.dataset_type))

            choices_list = list(expanded_choices_mapping.keys())

            prediction_trun = prediction.replace('\n', '。')
            prediction_sent = re.split('(。|！|\!|\.|？|\?)', prediction_trun)

            all_chosen_answers = []
            for sent in prediction_sent:
                sent = sent.strip("(。$,|！|\!|\.|？|\?)，；：？！…—·《》“”‘’{}[]（）()、|\\/\n\t\r\v\f ")
                if len(sent) == 1: sent = sent + " "
                if not sent: continue

                chosen_answers = [item for item in choices_list if sent.startswith(item) or sent.endswith(item)]
                if len(chosen_answers) > 1: chosen_answers = [chosen_answers[0]]

                all_chosen_answers.extend(chosen_answers)

            if len(all_chosen_answers) == 0:
                answer = "No Answer."

            else:
                chosen_labels = [expanded_choices_mapping[item] for item in all_chosen_answers]

                counter = Counter(chosen_labels)
                top2 = counter.most_common(2)

                if len(top2) == 1:
                    answer = top2[0][0]
                elif len(top2) == 2:
                    if top2[0][1] == top2[1][1]:
                        answer = "No Answer."
                    else:
                        answer = top2[0][0]

            if answer not in choices:
                answer = random.choice(choices)

            if answer == output:
                matching += 1
                match_one_by_one.append(1)
            else:
                match_one_by_one.append(0)
                

            all_predictioned_mapped.append(answer)
            
        accuracy = matching / len(data)

        return {'Accuracy': accuracy}
                

