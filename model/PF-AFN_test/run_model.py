from test import RunModel

def matching_filename(model, clothes):
    text = open('demo2.txt', 'w')
    text.write(model + ' ' + clothes)

def run(baseroot, output_path):
    RunModel().run_model(baseroot, output_path)