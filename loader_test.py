import data_loader

data_dir = 'data/poetry/'
input_file = 'tang(simplified).txt'
vocab_file = 'vocab_tang(simplified).pkl'
tensor_file = 'tensor_tang(simplified).npy'
data = data_loader.Data(data_dir, input_file, vocab_file, tensor_file)
