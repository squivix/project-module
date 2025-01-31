from candidate_extractors.TemplateMatchExtractor import TemplateMatchExtractor


def main():
    extractor = TemplateMatchExtractor()
    candidates = extractor.extract_candidates("data/whole-slides/gut/522021.svs")
    print(candidates)


main()
