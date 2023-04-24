import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset, make_dataset_together, make_dataset_from_folder
from PIL import Image, ImageOps
import glob

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        if opt.phase == 'train':
            dir_A = opt.dir_A
            self.dir_A = os.path.join(opt.dataroot, dir_A)
            dir_B = opt.dir_B
            self.dir_B = os.path.join(opt.dataroot, dir_B)
            
            self.A_paths, self.B_paths = make_dataset_together(self.dir_A, self.dir_B)
        elif opt.phase == 'test':
            if opt.phase_test_type == 'test_single':
                self.A_paths = [os.path.join(opt.dataroot, opt.dir_A)]
            elif opt.phase_test_type == 'test_all':
                self.A_paths = sorted(glob.glob(os.path.join(opt.dataroot, opt.dir_A, "*")))

        if opt.isTrain or opt.use_encoded_image:
            dir_B = opt.dir_B
            self.dir_B = os.path.join(opt.dataroot, dir_B)  
            self.B_paths = sorted(make_dataset(self.dir_B))
            
        self.A_paths = sorted(self.A_paths)
        self.dataset_size = len(self.A_paths) 
        print(self.dataset_size)
        
        if opt.phase == 'train':
            assert len(self.A_paths) == len(self.B_paths)

    def __getitem__(self, index):  
        grayscale_ = False
        if self.opt.output_nc == 1:
            grayscale_ = True      
        A_path = self.A_paths[index]              
        A = Image.open(A_path)        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))

        B_tensor = 0
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]   
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params, grayscale=grayscale_)      
            B_tensor = transform_B(B)

        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

class AlignedDataset_for_file(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        if opt.phase == 'train':
            f = open(os.path.join(opt.dataroot, opt.file_name_A), "r")
            all_data_A = f.readlines()
            f.close()

            f2 = open(os.path.join(opt.dataroot, opt.file_name_B), "r")
            all_data_B = f2.readlines()
            f2.close()

            self.A_paths = make_dataset_from_folder(all_data_A, opt.dataroot)
            self.B_paths = make_dataset_from_folder(all_data_B, opt.dataroot)
            
        elif opt.phase == 'test':
            if opt.phase_test_type == 'test_single':
                self.A_paths = [os.path.join(opt.dataroot, opt.dir_A)]
            elif opt.phase_test_type == 'test_all':
                self.A_paths = sorted(glob.glob(os.path.join(opt.dataroot, opt.dir_A, "*")))

        self.dataset_size = len(self.A_paths) 
        print(self.dataset_size)
        
        if opt.phase == 'train':
            assert len(self.A_paths) == len(self.B_paths)

    def __getitem__(self, index):  
        grayscale_ = False
        if self.opt.output_nc == 1:
            grayscale_ = True      
        A_path = self.A_paths[index]              
        A = Image.open(A_path.rstrip())        
        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        
        B_tensor = 0
        B_path = self.B_paths[index]
        B = Image.open(B_path.rstrip())
        params = get_params(self.opt, A.size)
        B_tensor = transform_A(B.convert('RGB'))

        input_dict = {'label': A_tensor, 'image': B_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'      
