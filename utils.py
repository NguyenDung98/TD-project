import re
import pandas as pd

link_pattern = r"https?\s*:\s*\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
dot_pattern = r"\s*\.+\s*(\.*\s*){100}"

def read_file(filename):
    data = open(filename, newline=None, encoding='utf-8')

    data.read()
    data.seek(0)
    lines = data.readlines()
    return lines

def get_list_stopwords(filename):
    lines = open(filename, newline=None, encoding='utf-8')

    lines.read()
    lines.seek(0)
    stopwords = lines.readlines()
    stopwords = list(map(lambda line: line.replace('\n', ''), stopwords))
    stopwords.pop(0)
    return stopwords

def pre_process_data(filename, stopwords_filename):
    lines = read_file(filename)
    lowercaseLines = list(map(lambda line: line.lower(), lines))
    linesWithoutLinks = list(map(lambda line: re.sub(link_pattern, '', line), lowercaseLines))
    linesFormattedDot = list(map(lambda line: re.sub(dot_pattern, ' punc ', line), linesWithoutLinks))
    linesWithoutWierdChar = list(map(lambda line: re.sub(r'[\d\W]+', ' ', line), linesFormattedDot))
    formattedLines = list(map(lambda line: line.replace('\n', '').split(" ", 1), linesWithoutWierdChar))
    
    df = pd.DataFrame(formattedLines, columns=['label', 'content'])
    datalist = df['content'].tolist()
    linesWithoutUnderScores = list((map(lambda line: re.sub(r'_+', ' ', line), datalist)))
    removedOneWordLines = list((map(lambda line: re.sub(r'\s+\w\s+', " ",line), linesWithoutUnderScores)))
    stopwords = get_list_stopwords(stopwords_filename)
    linesWithoutStopwords = list(map(lambda line: " ".join(list(filter(lambda word: stopwords.count(word) == 0, line.split(" ")))), removedOneWordLines))
    df['content'] = linesWithoutStopwords
    return df

def pre_process_test_data(filename, stopwords_filename):
    lines = read_file(filename)
    lowercaseLines = list(map(lambda line: line.lower(), lines))
    linesWithoutLinks = list(map(lambda line: re.sub(link_pattern, '', line), lowercaseLines))
    linesFormattedDot = list(map(lambda line: re.sub(dot_pattern, ' punc ', line), linesWithoutLinks))
    linesWithoutWierdChar = list(map(lambda line: re.sub(r'[\d\W]+', ' ', line), linesFormattedDot))
    
    df = pd.DataFrame(linesWithoutWierdChar, columns=['content'])
    datalist = df['content'].tolist()
    linesWithoutUnderScores = list((map(lambda line: re.sub(r'_+', ' ', line), datalist)))
    removedOneWordLines = list((map(lambda line: re.sub(r'\s+\w\s+', " ",line), linesWithoutUnderScores)))
    stopwords = get_list_stopwords(stopwords_filename)
    linesWithoutStopwords = list(map(lambda line: " ".join(list(filter(lambda word: stopwords.count(word) == 0, line.split(" ")))), removedOneWordLines))
    df['content'] = linesWithoutStopwords
    return df

def store_file(output_file, features, labels):
    with open(output_file, 'w', newline='\n', encoding='utf-8') as f:
        for i in range(0, len(features)):
            if labels[i] == '':
                f.write("%s\n" % (features[i]))
            else:
                f.write("%s %s\n" % (labels[i], features[i]))    