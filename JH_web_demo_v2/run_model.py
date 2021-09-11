from PFAFN.test import RunModel

def matching_filename(model, clothes):
    text = open('PFAFN/demo.txt', 'w')
    text.write(model + ' ' + clothes)
    return text

def run(baseroot, output_path):
    RunModel().run_model(baseroot, output_path)