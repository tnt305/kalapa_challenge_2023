import pandas as pd
import unicodedata
import regex as re
from pyvi import ViTokenizer, ViPosTagger
import string

def clean_answer(answer):
    l_cleaned = ["A:", "B:", "C:", "D:", "E:", "F:", "A.", "B.", "C.", "D.", "E.", "F."]
    for item in l_cleaned:
        answer = answer.replace(item, "")
    answer = " ".join(answer.split())
    return answer


bang_nguyen_am= [['a', 'à', 'á', 'ả', 'ã', 'ạ', 'a'],
                  ['ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'aw'],
                  ['â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'aa'],
                  ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'e'],
                  ['ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'ee'],
                  ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị', 'i'],
                  ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'o'],
                  ['ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'oo'],
                  ['ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'ow'],
                  ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'u'],
                  ['ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'uw'],
                  ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ', 'y']]

bang_ky_tu_dau = ['', 'f', 's', 'r', 'x', 'j']
nguyen_am_to_ids = {}

for i in range(len(bang_nguyen_am)):
    for j in range(len(bang_nguyen_am[i]) - 1):
        nguyen_am_to_ids[bang_nguyen_am[i][j]] = (i, j)

# Chuẩn hóa unicode
# Có 2 loại unicode : unicode tổ hơp và unicode dựng sẵn, điêu này dẫn tới việc 2 từ giống nhau sẽ bị coi là khác nhau
# Chuẩn hóa tất cả về 1 loại là unicode dựng sẵn
def chuan_hoa_unicode(text):
	text = unicodedata.normalize('NFC', text)
	return text

dict_map = {
    "òa": "oà",
    "Òa": "Oà",
    "ÒA": "OÀ",
    "óa": "oá",
    "Óa": "Oá",
    "ÓA": "OÁ",
    "ỏa": "oả",
    "Ỏa": "Oả",
    "ỎA": "OẢ",
    "õa": "oã",
    "Õa": "Oã",
    "ÕA": "OÃ",
    "ọa": "oạ",
    "Ọa": "Oạ",
    "ỌA": "OẠ",
    "òe": "oè",
    "Òe": "Oè",
    "ÒE": "OÈ",
    "óe": "oé",
    "Óe": "Oé",
    "ÓE": "OÉ",
    "ỏe": "oẻ",
    "Ỏe": "Oẻ",
    "ỎE": "OẺ",
    "õe": "oẽ",
    "Õe": "Oẽ",
    "ÕE": "OẼ",
    "ọe": "oẹ",
    "Ọe": "Oẹ",
    "ỌE": "OẸ",
    "ùy": "uỳ",
    "Ùy": "Uỳ",
    "ÙY": "UỲ",
    "úy": "uý",
    "Úy": "Uý",
    "ÚY": "UÝ",
    "ủy": "uỷ",
    "Ủy": "Uỷ",
    "ỦY": "UỶ",
    "ũy": "uỹ",
    "Ũy": "Uỹ",
    "ŨY": "UỸ",
    "ụy": "uỵ",
    "Ụy": "Uỵ",
    "ỤY": "UỴ",
    }

def replace_all(text, dict_map):
    for i, j in dict_map.items():
        text = text.replace(i, j)
    return text

# Có 2 kiểu gõ dấu ở Tiếng Việt, ví dụ như là : òa hoặc oà (ta gọi lần lượt là chuẩn 1 và 2). Mặc dù kiểu gõ chữ sau ít
#phổ biến hơn tuy nhiên vẫn cần phải chuẩn hóa tránh việc một số văn bản vẫn sử dụng kiểu gõ dấu thứ 2.
"""
	Hàm này xử lý chuẩn hóa từng từ một, sau khi chuẩn hóa từng từ thì ta sẽ đi chuân hóa từng câu sau
	"""
def chuan_hoa_dau_tu_tieng_viet(word):
    if not is_valid_vietnam_word(word):
        return word

    chars = list(word)
    dau_cau = 0
    nguyen_am_index = []
    qu_or_gi = False
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x == -1:
            continue
        elif x == 9:  # check qu
            if index != 0 and chars[index - 1] == 'q':
                chars[index] = 'u'
                qu_or_gi = True
        elif x == 5:  # check gi
            if index != 0 and chars[index - 1] == 'g':
                chars[index] = 'i'
                qu_or_gi = True
        if y != 0:
            dau_cau = y
            chars[index] = bang_nguyen_am[x][0]
        if not qu_or_gi or index != 1:
            nguyen_am_index.append(index)
    if len(nguyen_am_index) < 2:
        if qu_or_gi:
            if len(chars) == 2:
                x, y = nguyen_am_to_ids.get(chars[1])
                chars[1] = bang_nguyen_am[x][dau_cau]
            else:
                x, y = nguyen_am_to_ids.get(chars[2], (-1, -1))
                if x != -1:
                    chars[2] = bang_nguyen_am[x][dau_cau]
                else:
                    chars[1] = bang_nguyen_am[5][dau_cau] if chars[1] == 'i' else bang_nguyen_am[9][dau_cau]
            return ''.join(chars)
        return word

    for index in nguyen_am_index:
        x, y = nguyen_am_to_ids[chars[index]]
        if x == 4 or x == 8:  # ê, ơ
            chars[index] = bang_nguyen_am[x][dau_cau]
            # for index2 in nguyen_am_index:
            #     if index2 != index:
            #         x, y = nguyen_am_to_ids[chars[index]]
            #         chars[index2] = bang_nguyen_am[x][0]
            return ''.join(chars)

    if len(nguyen_am_index) == 2:
        if nguyen_am_index[-1] == len(chars) - 1:
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            chars[nguyen_am_index[0]] = bang_nguyen_am[x][dau_cau]
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            # chars[nguyen_am_index[1]] = bang_nguyen_am[x][0]
        else:
            # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
            # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
            x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
            chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
    else:
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[0]]]
        # chars[nguyen_am_index[0]] = bang_nguyen_am[x][0]
        x, y = nguyen_am_to_ids[chars[nguyen_am_index[1]]]
        chars[nguyen_am_index[1]] = bang_nguyen_am[x][dau_cau]
        # x, y = nguyen_am_to_ids[chars[nguyen_am_index[2]]]
        # chars[nguyen_am_index[2]] = bang_nguyen_am[x][0]
    return ''.join(chars)


def is_valid_vietnam_word(word):
    chars = list(word)
    nguyen_am_index = -1
    for index, char in enumerate(chars):
        x, y = nguyen_am_to_ids.get(char, (-1, -1))
        if x != -1:
            if nguyen_am_index == -1:
                nguyen_am_index = index
            else:
                if index - nguyen_am_index != 1:
                    return False
                nguyen_am_index = index
    return True

def chuan_hoa_dau_cau_tieng_viet(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    for index, word in enumerate(words):
        cw = re.sub(r'(^[\p{P}]*)([\p{L}.]*\p{L}+)([\p{P}]*$)', r'\1/\2/\3', word).split('/')
        if len(cw) == 3:
            cw[1] = chuan_hoa_dau_tu_tieng_viet(cw[1])  # You should ensure that chuan_hoa_dau_tu_tieng_viet is defined.
        words[index] = ''.join(cw)
    return ' '.join(words)

# Tách từ tiếng việt, từ tiếng việt không giống như tiếng anh, tách từ tiếng anh ta chỉ cần tách bằng khoảng trắng
# Tuy nhiên từ tiếng Việt có cả từ đơn lẫn từ ghép nên tách từ tiêng Việt sẽ phúc tạp hơn
# Project sử dung thu viện pyvi (xem mã nguồn tại : https://github.com/trungtv/pyvi) để phục vụ bài toán con tách từ Tiếng Việt
def tach_tu_tieng_viet(text):
	text = ViTokenizer.tokenize(text)
	return text

# Đưa về chữ viết thường
def chuyen_chu_thuong(text):
	return text.lower()

# Xóa đi các dấu cách thừa, các từ không cần thiết cho việc phân loại vẳn bản
def chuan_hoa_cau(text):
	text = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text

def tien_xu_li(text):
	text = chuan_hoa_unicode(text)
	text = chuan_hoa_dau_cau_tieng_viet(text)
	# text = tach_tu_tieng_viet(text)
	text = chuyen_chu_thuong(text)
	text = chuan_hoa_cau(text)

	return text



def remove_html(text):
    text = re.sub(r'<[^>]+>','', text)
    text = re.sub(r'[\n]','', text)
    text =  text.strip()
    return text
def clean_text(text):
    return re.sub(r'[\n]', '', text).strip()
def remove_punc_title(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
        text = re.sub(r'\s+', ' ', text)
    return text

def remove_url(text):
    return re.sub(r"http\S+", "", text)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

#codes = ["–", "&", "&#8211", "#8211"]
def remove_special_token(text):
    codes = ["–", "&"]
    pattern1 = re.compile(r'<iframe.*?="')
    text = re.sub(pattern1, '', text)

    pattern2 = re.compile(r'&#[0-9]+;')
    text = re.sub(pattern2, '', text).strip()

    text = re.sub(r'\s+', ' ', text)
    for code in codes:
        text = text.replace(code, " ")
    return text

def remove_punctation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ')
    return text

def preprocess_text(text, remove_punc=True, is_lower=True):
    if remove_punc:
        text = " ".join(remove_punctation(remove_special_token(remove_html_tags(remove_url(text)))).split())
    else:
        text = " ".join(remove_special_token(remove_html_tags(remove_url(text))).split())
    if is_lower:
        return text.lower()
    else:
        return text

codes = ["–", "&"]
def get_titles(file_names, corpus):
    titles = []
    for file_name, docs in zip(file_names, corpus):
        doc = docs.split("\n")
        for item in doc:
            match = re.search(r'<h1>(.*?)</h1>', item)
            if match:
                h1_content = match.group(1)
                break
            if not match:
                h1_content = doc[1]
        h1_content = h1_content.split(":")[0].split("?")[0]
        for code in codes:
            h1_content = h1_content.replace(code, " ")
        h1_content = preprocess_text(h1_content)
        titles.append(h1_content)
    return titles

def corpus_rename(corpus):
    for para in corpus:
        if "hệ thống bệnh viện đa khoa tâm anh" in para.lower():
            process_paragraph = para.lower().split("hệ thống bệnh viện đa khoa tâm anh")[0]
            process_paragraph = para[ : len(process_paragraph)]
        else:
            process_paragraph = para
        corpus[corpus.index(para)] = process_paragraph
    return corpus

def process_document(title, document):
    doc_dict = {'title': '', 'content': '', 'sub_title': [], 'item': []}
    pattern = None

    if 'Mục lục' in document:
        paragraphs = document.split(" CHUYÊN MỤC BỆNH HỌC >")[1]
        paragraphs_title, paragraphs_res = re.split('<h1>|</h1>', paragraphs, 1)
        doc_dict['title'] = remove_punc_title(clean_text(paragraphs_title))

        paragraphs_content, paragraphs_res = paragraphs_res.split('<h3>Mục lục</h3>', 1)
        doc_dict['content'] = remove_html(paragraphs_content)

        paragraphs_res = paragraphs_res.split('\n <h2>')
        sub_title = [item.strip() for item in paragraphs_res[0].split('\n') if item != '']
        doc_dict['sub_title'] = sub_title

        content_items = [preprocess_text(paragraphs_res[i], remove_punc=False, is_lower=False).strip() for i in range(1, len(paragraphs_res)) if paragraphs_res[i] != '']
        doc_dict['item'] = content_items
    else:
        if title == 'ốm nghén khi mang thai':
            pattern = ['Ốm nghén là gì?', 'Triệu chứng của cơn nghén', 'Ốm nghén có tốt không? Có ảnh hưởng thai nhi không?',
                       'Nguyên nhân gây ốm nghén khi mang thai', 'Phương pháp chẩn đoán cơn nghén', 'Kiểm soát cơn nghén bầu',
                       'Thuốc hỗ trợ cải thiện ốm nghén']
        if title == 'hội chứng truyền máu song thai':
            pattern = ['Truyền máu song thai là gì?', 'Nguyên nhân gây truyền máu song thai', 'Triệu chứng truyền máu song thai',
                       'Đối tượng nguy cơ cao mắc bệnh', 'Phương pháp chẩn đoán truyền máu song nhi', 'Phương pháp điều trị truyền máu thai đôi',
                       'Biện pháp phòng ngừa truyền máu song thai']

        doc_dict['title'] = remove_punc_title(document.split('\n')[1])
        doc_dict['content'] = document.split('\n')[2]
        pattern = pattern

        split_pattern = r'({})'.format('|'.join(map(re.escape, pattern)))
        list_item = re.split(split_pattern, document)
        sub_title = preprocess_text(','.join(list_item[1::2]), remove_punc=False, is_lower=False)
        doc_dict['sub_title'] = sub_title
        doc_dict['item'] = [preprocess_text(item, False, False) for item in list_item[::2]]

    return doc_dict

def prepare_sentence_for_encode(doc):
    sub_corpus = []
    title = doc['title']
    content = doc['content']
    items = doc['item']
    sentence_headlines = title + ". " + content
    sub_corpus.append(sentence_headlines)
    for idx in range(len(items)):
        sentences =  title + ". "+ items[idx]
        sentences = remove_html_tags(remove_url(sentences))
        sentences = " ".join(sentences.split())
        sub_corpus.append(sentences)
    return sub_corpus
