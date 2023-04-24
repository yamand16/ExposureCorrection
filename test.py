import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import time
import numpy as np

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)
            
    if opt.verbose:
        print(model)
else:
    from run_engine import run_trt_engine, run_onnx
    
print("total parameters: {}\n" .format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    if opt.data_type == 16:
        data['label'] = data['label'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
    if opt.export_onnx:
        print ("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label']],
                          opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1 
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label']])
    else:        
        generated = model.inference(data['label'], data['image'])
        

        
    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], 0)),
                           ('synthesized_image', util.tensor2im(generated.data[0]))])

    if opt.batchSize > 1:
        for index_for_img_save in range(opt.batchSize):
            img_path = data['path'][index_for_img_save]
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][index_for_img_save], 0)),
                                    ('synthesized_image', util.tensor2im(generated.data[index_for_img_save]))])
            print('process image... %s' % img_path)
            visualizer.save_images(webpage, visuals, [img_path])
    else:
        img_path = data['path']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

webpage.save()
