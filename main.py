from filemonitor import FileMonitor
from sentence2class import SentenceParser
from setup import SetupGloveDict
from imageGen import ImageGen
import time
import sys
import json

if __name__ == '__main__':

    with open('categories.json', 'r') as f:
        categories = json.load(f)
        f.close()

    sentence_parser = SentenceParser("./categories_dict.json", categories)

    monitor = FileMonitor('unity_text.txt')
    img_gen = ImageGen()
    try:
        print("File monitoring started...")
        while True:
            if monitor.filechange():
                nouns = monitor.extract_nouns()
                print(nouns)

                class_nums = sentence_parser.getClassesFromSentence(nouns)
                print(class_nums)
                img_gen.generate_image(class_nums)

            # extract keywords from sentence using Alex's code
            time.sleep(0.2)
    except KeyboardInterrupt:
        sys.exit()
