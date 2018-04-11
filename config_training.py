config = {'train_data_path':['/data/LUNA16/subset0/',
                             '/data/LUNA16/subset1/',
                             '/data/LUNA16/subset2/',
                             '/data/LUNA16/subset3/',
                             '/data/LUNA16/subset4/',
                             '/data/LUNA16/subset5/',
                             '/data/LUNA16/subset6/',
                             '/data/LUNA16/subset7/',
                             '/data/LUNA16/subset8/'],
          'val_data_path':['/data/LUNA16/subset8/'],
          'test_data_path':['/data/LUNA16/subset9/'],
          
          'train_preprocess_result_path':'/data/preprocess/luna_preprocess_subset/',
          'val_preprocess_result_path':'/data/preprocess/luna_preprocess_subset/',
          'test_preprocess_result_path':'/data/preprocess/luna_preprocess_subset/',
          
          'train_annos_path':'/data/LUNA16/CSVFILES/annotations.csv',
          'val_annos_path':'/data/LUNA16/CSVFILES/annotations.csv',
          'test_annos_path':'/data/LUNA16/CSVFILES/annotations.csv',

          'black_list':[],
          
          'preprocessing_backend':'python',

          'luna_segment':'/data/LUNA16/seg-lungs-LUNA16/',
          'preprocess_result_path':'preprocess/',
          'luna_data':'/data/LUNA16/',
          'luna_label':'/data/LUNA16/CSVFILES/annotations.csv'
         } 