import numpy as np
import eraser

class EraserEvaluation:
    def __init__(self, eraser, tokenizer):
        self.eraser = eraser
        self.tokenizer = tokenizer

    # shap_values.data[0]
    def map_tokens(self, tokens, doc_tokens):
        tokens = np.array(tokens)
        i = 0
        j = 0
        n = tokens.size
        m = len(doc_tokens)
        tokens_map = []

        while i < n and j < m:
            model_token = tokens[i].strip()

            if model_token == doc_tokens[j]:
                tokens_map.append(j)
                j += 1
            elif model_token == "":
                tokens_map.append(None)
            else:
                for k in range(1, 5):
                    if (i + k < n):
                        merged_token = model_token

                        for l in range(1, k + 1):
                            merged_token += tokens[i + l].strip()

                        if merged_token == doc_tokens[j]:
                            for l in range(1, k + 2):
                                tokens_map.append(j)

                            j += 1
                            i += k
                            break

            i += 1

        return tokens_map

    def merge_owen_for_tokens(self, owen_values, tokens_map):
        new_owen = [owen_values[0]]
        last_position = 0

        for i in range(1, len(tokens_map)):
            if (tokens_map[i-1] is None) or (tokens_map[i-1] == tokens_map[i]):
                new_owen[last_position] += owen_values[i]
            else:
                last_position += 1
                new_owen.append(owen_values[i])

        return new_owen

    def prepare_truth_pred_dicts(self, pred_owens, evidences):
        pred = {}
        truth = {}
        evidences_tokens = []
        len_tokens = len(pred_owens)

        for ev in evidences:
            if ev.start_token <= len_tokens:
                start_token = ev.start_token

                if ev.end_token <= len_tokens - 1:
                    end_token = ev.end_token
                else:
                    end_token = len_tokens - 1

                evidences_tokens += list(range(start_token, end_token + 1))

        for i in range(len(pred_owens)):
            pred[i] = pred_owens[i]

            if i in evidences_tokens:
                truth[i] = True
            else:
                truth[i] = False

        return truth, pred

    def prepare_truth_pred_lists(self, pred_owens, evidences):
        evidences_tokens = []
        len_tokens = len(pred_owens)

        for ev in evidences:
            if ev.start_token <= len_tokens:
                # start_token = tokens_map_reversed[ev.start_token][0]
                start_token = ev.start_token

                if ev.end_token <= len_tokens - 1:
                    end_token = ev.end_token
                else:
                    end_token = len_tokens - 1

                evidences_tokens += list(range(start_token, end_token + 1))

        truth = [(i in evidences_tokens) for i in range(len(pred_owens))]
        return truth, pred_owens

    def docs_auprc_mean(self):
        docs_auc = []

        for i in range(1, 50):
            print(i)
            annotation = test[i]
            evidences = annotation.all_evidences()
            (docid,) = set(ev.docid for ev in evidences)
            doc = self.eraser.documents[docid]
            sentences = []

            for sent in doc:
                sentence = ' '.join(sent)
                sentences.append(sentence)

            doc_text = ' '.join(sentences)
            total_tokens = self.tokenizer(doc_text)
            n_tokens = len(total_tokens['input_ids'])
            n_sents = len(sentences)

            if n_tokens > 512:
                rel_tokens = 512 / n_tokens
                n_sents = int(rel_tokens * len(sentences)) - 1
                doc_text1 = ' '.join(sentences[:n_sents])
                tokens1 = self.tokenizer(doc_text1)

                while len(tokens1['input_ids']) > 512:
                    n_sents -= 1
                    doc_text1 = ' '.join(sentences[:n_sents])
                    tokens1 = self.tokenizer(doc_text1)
            # else:
            #     doc_text1 = doc_text
            #     tokens1 = total_tokens

            doc_tokens = []

            i_sent = 0

            for sent in doc:
                if i_sent < n_sents:
                    doc_tokens += sent
                    i_sent += 1

            shap_values = docs_owen[docid]
            tokens_map = map_tokens(shap_values, doc_tokens)

            if annotation.classification == "NEG":
                label_num = 2
            else:
                label_num = 0

            owen_values = shap_values.values[0, :, label_num]
            pred_owens = self.merge_owen_for_tokens(owen_values, tokens_map)
            truth, pred = self.prepare_truth_pred_lists(pred_owens, evidences)
            aucs = []
            precision, recall, _ = precision_recall_curve(truth, pred)
            auprc = auc(recall, precision)
            # aucs.append(auc(recall, precision))
            # doc_auc = np.average(aucs)
            # print("auprc" , doc_auc)
            print("auprc ", auprc)
            docs_auc.append(auprc)

        return np.mean(docs_auc)

    def get_doc_owens(i, documents, self):
        annotation = test[i]
        evidences = annotation.all_evidences()
        (docid,) = set(ev.docid for ev in evidences)
        doc = documents[docid]
        sentences = []

        for sent in doc:
            sentence = ' '.join(sent)
            sentences.append(sentence)

        doc_text = ' '.join(sentences)
        total_tokens = tokenizer(doc_text)
        n_tokens = len(total_tokens['input_ids'])
        n_sents = len(sentences)

        if n_tokens > 512:
            rel_tokens = 512 / n_tokens
            n_sents = int(rel_tokens * len(sentences)) - 1
            doc_text1 = ' '.join(sentences[:n_sents])
            tokens1 = tokenizer(doc_text1)

            while len(tokens1['input_ids']) > 512:
                n_sents -= 1
                doc_text1 = ' '.join(sentences[:n_sents])
                tokens1 = tokenizer(doc_text1)
        else:
            doc_text1 = doc_text
            tokens1 = total_tokens

        doc_tokens = []

        i_sent = 0

        for sent in doc:
            if i_sent < n_sents:
                doc_tokens += sent
                i_sent += 1

        shap_values = docs_owen[docid]
        tokens_map = map_tokens(shap_values, doc_tokens)

        if annotation.classification == "NEG":
            label_num = 2
        else:
            label_num = 0

        owen_values = shap_values.values[0, :, label_num]
        return owen_values, evidences, tokens_map, doc_tokens

    def evaluate(self, feature_importance_scores, test, documents, tokenizer):
        docs_auc = []
        all_truth = []
        all_preds = []

        for i in range(1, 50):
            print(i)
            annotation = test[i]
            evidences = annotation.all_evidences()
            (docid,) = set(ev.docid for ev in evidences)
            doc = documents[docid]
            sentences = []

            for sent in doc:
                sentence = ' '.join(sent)
                sentences.append(sentence)

            doc_text = ' '.join(sentences)
            total_tokens = tokenizer(doc_text)
            n_tokens = len(total_tokens['input_ids'])
            n_sents = len(sentences)

            if n_tokens > 512:
                rel_tokens = 512 / n_tokens
                n_sents = int(rel_tokens * len(sentences)) - 1
                doc_text1 = ' '.join(sentences[:n_sents])
                tokens1 = tokenizer(doc_text1)

                while len(tokens1['input_ids']) > 512:
                    n_sents -= 1
                    doc_text1 = ' '.join(sentences[:n_sents])
                    tokens1 = tokenizer(doc_text1)
            # else:
            #     doc_text1 = doc_text
            #     tokens1 = total_tokens

            doc_tokens = []

            i_sent = 0

            for sent in doc:
                if i_sent < n_sents:
                    doc_tokens += sent
                    i_sent += 1

            # file_pi = open('test_docs_owen_x.pkl', 'rb')
            # docs_owen = pickle.load(file_pi)
            # shap_values = docs_owen[docid]
            doc_features_importance = feature_importance_scores[docid]
            tokens_map = self.map_tokens(doc_features_importance, doc_tokens)

            if annotation.classification == "NEG":
                label_num = 2
            else:
                label_num = 0

            owen_values = shap_values.values[0, :, label_num]
            pred_owens = merge_owen_for_tokens(owen_values, tokens_map)
            truth, pred = prepare_truth_pred_lists(pred_owens, evidences)
            aucs = []
            precision, recall, _ = precision_recall_curve(truth, pred)
            auprc = auc(recall, precision)
            # aucs.append(auc(recall, precision))
            # doc_auc = np.average(aucs)
            # print("auprc" , doc_auc)
            print("auprc ", auprc)
            docs_auc.append(auprc)

            ######################
            owen_values, evidences, tokens_map, doc_tokens = get_doc_owens(i)
            pred_owens = merge_owen_for_tokens(owen_values, tokens_map)
            truth, pred = prepare_truth_pred_lists(pred_owens, evidences)
            all_truth += truth
            all_preds += pred


        auprc1 = np.mean(docs_auc)
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc2 = auc(recall, precision)

        return auprc1, auprc2