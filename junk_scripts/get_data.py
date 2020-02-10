def get_data(set):
    source = "../en-de/%s.ende.src" % set
    mt = "../en-de/%s.ende.mt" % set
    scores = "../en-de/%s.ende.scores" % set
    with open(source, "r", encoding='utf-8') as source, open(mt, "r", encoding='utf-8') as mt, open(scores, "r", encoding='utf-8') as scores:
        return list(zip(source.readlines(), mt.readlines(), scores.readlines()))

print(get_data("dev"))

