import json
import shutil



def main():

    anns  =[]
    with open('Jakub_lesions.json') as f:
        data = json.load(f)
    for item in data:
        anns.append(Annotation(item['id'], item["annotations"][0]["id"], item["file_upload"].split("-")[-1]))
  
    for ann in anns:
        # copy file to new folder with another name
        try:
            shutil.copyfile(f'./Jakub_lesions/{ann.get_annotation()}', f'./masks/{ann.filename}_mask.npy')
        except:
            print("File not found: ", ann.get_annotation())
            continue


class Annotation:
    main_id = ""
    second_id = ""
    brush_name="lesions"
    filename = ""

    def __init__(self, main_id, second_id, filename):
        self.main_id = main_id
        self.second_id = second_id
        self.filename =filename.split(".")[0]
    def __str__(self):
        return str(self.main_id) + " " + str(self.second_id) + " " + self.filename
    
    def get_annotation(self):
        return f'task-{self.main_id}-annotation-{self.second_id}-by-1-tag-{self.brush_name}-0.npy'

    



if __name__ == '__main__':
    main()
    # task-33-annotation-18-by-1-tag-lesion-0.npy