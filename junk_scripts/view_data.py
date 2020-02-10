def print_data(set):
    source = "../en-de/%s.ende.src" % set
    mt = "../en-de/%s.ende.mt" % set
    scores = "../en-de/%s.ende.scores" % set
    with open(source, "rb") as source, open(mt, "rb") as mt, open(scores, "rb") as scores:
        for a, b, c in zip(source.readlines(), mt.readlines(), scores.readlines()):
            print(a, b, c)

print_data("dev")