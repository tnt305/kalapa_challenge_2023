from retrieval_based.utils import * 

def calculate_cosince_similarity(text_1, text_2):
    text_1 = tien_xu_li(text_1)
    text_2 = tien_xu_li(text_2)
    sentences_token_1 = [tokenize(sentence) for sentence in [text_1]]
    embedding_1 = model.encode(sentences_token_1)
    embedding_1 = torch.from_numpy(embedding_1)
    sentences_token_2 = [tokenize(sentence) for sentence in [text_2]]
    embedding_2 = model.encode(sentences_token_2)
    embedding_2 = torch.from_numpy(embedding_2)
    rs = torch.cosine_similarity(embedding_1.reshape(1,-1), embedding_2.reshape(1,-1))
    rs = rs.numpy()[0]
    return rs

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def select_top_1_answer(answer_per_question, question, relavant_corpus):
    l_prediction = ['0' for item in answer_per_question]
    l_cosine_scores = []
    for answer in answer_per_question:
        question = ' '.join(tokenize(question).split()[:-2])
        cosine_score = calculate_cosince_similarity(question + " " + answer, relavant_corpus)
        l_cosine_scores.append(cosine_score)

    idx_seletect = np.argmax(np.array(l_cosine_scores))
    l_prediction[idx_seletect] = '1'

    return l_prediction

def get_top_1_embedding(question, l_raw_sentences_matched):
    index_faiss = faiss.IndexFlatL2(768)
    sentences = [tien_xu_li(item) for item in l_raw_sentences_matched]
    sentences_token = [tokenize(sentence) for sentence in sentences]
    embedding = model.encode(sentences_token)
    embedding = np.array(embedding).astype('float32')
    index_faiss.add(embedding)

    sentences_token_qa = [tokenize(sentence) for sentence in [question]]
    embedding_qa = model.encode(sentences_token_qa)[0]
    embedding_qa = np.array([embedding_qa]).astype('float32')


    f_dists, f_ids = index_faiss.search(embedding_qa.reshape(1, -1), k= 5)
    print(f_dists)
    print('---')
    print(f_ids)
    return f_ids[0][0]

def get_predict_1_answer(answer_per_question, question, relavant_corpus):
    l_prediction = ['0' for item in answer_per_question]
    l_cosine_scores = []
    for answer in answer_per_question:
        cosine_score = calculate_cosince_similarity(answer, relavant_corpus)
        l_cosine_scores.append(cosine_score)
    idx_seletect = np.argmax(np.array(l_cosine_scores))
    l_prediction[idx_seletect] = '1'
    prediction_str = "".join(l_prediction)
    return prediction_str

def get_predict_true_false(answer_per_question, question, relavant_corpus):
    l_prediction = ['0' for item in answer_per_question]
    cosine_score = calculate_cosince_similarity(question, relavant_corpus)
    if cosine_score >= 0.6:
        l_prediction[0] = '1'
    else:
        l_prediction[-1] = '1'
    prediction_str = "".join(l_prediction)
    return prediction_str

def get_predict_multi_answer(answer_per_question, question, relavant_corpus):
    THRESHOLD = 0.6
    l_prediction = ['0' for item in answer_per_question]
    found_matched = False
    if "Có" not in answer_per_question and "có" not in answer_per_question:
        #change other method
        for idx_answer, answer in enumerate(answer_per_question):
            tmp_answer = vncorenlp.tokenize(answer)
            keywords_answers = []
            for sentence in tmp_answer:
                keywords_answers += [word.lower() for word in sentence]
            #split relavant corpus
            keywords_corpus = []
            tmp_corpus = vncorenlp.tokenize(relavant_corpus)
            for sentence in tmp_corpus:
                keywords_corpus += [word.lower() for word in sentence]
            l_overlap = intersection(keywords_answers, keywords_corpus)
            ratio_overlap = len(l_overlap) / len(keywords_answers)
            if ratio_overlap >= THRESHOLD:
                l_prediction[idx_answer] = '1'
                found_matched = True
    if not found_matched:
        l_prediction = select_top_1_answer(answer_per_question, question, relavant_corpus)
    prediction_str = "".join(l_prediction)
    return prediction_str

def get_predict(answer_per_question, question, relavant_corpus):
    l_1_answers = ["là gì", "cách gì", 'bệnh gì', 'bệnh nào']
    l_true_false = ["đúng hay sai", "hay không"]
    type_predict = 3
    if "là gì" in question and "các" not in question.lower():
        type_predict = 1
    if "cách gì" in question and "các" not in question.lower():
        type_predict = 1
    if 'bệnh gì' in question.lower() and 'những' not in question.lower():
        type_predict = 1
    if 'bệnh nào' in question.lower() and 'những' not in question.lower():
        type_predict = 1
    if 'bệnh lý nào' in question.lower():
        type_predict = 1
    if 'định nghĩa' in question.lower():
        type_predict = 1
    if "đúng hay sai" in question.lower():
        type_predict = 2
    if "hay không" in question.lower():
        type_predict = 2

    if type_predict == 1:
        prediction_str = get_predict_1_answer(answer_per_question, question, relavant_corpus)
    elif type_predict == 2:
        prediction_str = get_predict_true_false(answer_per_question, question, relavant_corpus)
    else:
        prediction_str = get_predict_multi_answer(answer_per_question, question, relavant_corpus)
    return prediction_str

def find_relavant_in_current_document(idx_matched_info, l_infos, question, answer_per_question):
    index_faiss = faiss.IndexFlatL2(768)
    info = l_infos[idx_matched_info]
    subcorpus = prepare_sentence_for_encode(info)
    print('find subcorpus ', subcorpus)
    sentences = [tien_xu_li(item) for item in subcorpus]
    print('find ssentences ', sentences)
    sentences_token = [tokenize(sentence) for sentence in sentences]
    embedding = model.encode(sentences_token)
    embedding = np.array(embedding).astype('float32')
    index_faiss.add(embedding)


    question = ' '.join(question.split()[:-2])
    question_with_answer = question
    for answer in answer_per_question:
        question_with_answer += " " + answer
    question_with_answer = tien_xu_li(question_with_answer)
    print('find q/w_ans ', question_with_answer)
    sentences_token_qa = [tokenize(sentence) for sentence in [question_with_answer]]
    embedding_qa = model.encode(sentences_token_qa)[0]
    embedding_qa = np.array([embedding_qa]).astype('float32')
    f_dists, f_ids = index_faiss.search(embedding_qa.reshape(1, -1), k=1)
    print('find f_dists ', f_dists)
    print('find f_ids ', f_ids)
    print('******')
    return subcorpus[f_ids[0][0]]