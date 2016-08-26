from six.moves import cPickle as pickle
import numpy as np

class dataset:
    def __init__(self,file_name):
        self.train_dataset,self.train_labels,self.valid_dataset,self.valid_labels,self.test_dataset,self.test_labels=self.read_data(file_name)
        self.show_shape()

        self.set_data()
        self.show_shape()

        self.length=len(self.train_labels)
        self.index_epoch=0
        self.epochs_completed=0


    def read_data(self,pickle_file):
        f = file(pickle_file, 'rb')
        a = pickle.load(f)
        f.close()
        return a.get('train_dataset'),a.get('train_labels'),a.get('valid_dataset'),a.get('valid_labels'),a.get('test_dataset'),a.get('test_labels')

    def set_data(self):
        self.train_dataset,self.train_labels=self.reform_data(self.train_dataset,self.train_labels)
        self.valid_dataset, self.valid_labels = self.reform_data(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reform_data(self.test_dataset, self.test_labels)

    def reform_data(self,data,label):
        image_size = 28
        num_labels = 10
        data = data.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        label = (np.arange(num_labels) == label[:, None]).astype(np.float32)
        return data, label

    def show_shape(self):
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)

    def set_length(self,sub_set):
        self.length=sub_set
        self.train_dataset=self.train_dataset[0:sub_set]
        self.train_labels=self.train_labels[0:sub_set]

    def next_batch(self,batch_size):
        start=self.index_epoch
        self.index_epoch+=batch_size
        if self.index_epoch>self.length:
            #Finish epoch once
            self.epochs_completed+=1
            #Shuffle the data again
            perm = np.arange(0, self.length)
            np.random.shuffle(perm)
            # print (perm.shape)
            # print type(perm)
            self.train_dataset = self.train_dataset[perm]
            self.train_labels = self.train_labels[perm]
            #start next epoch
            start=0
            self.index_epoch=batch_size
            assert batch_size<=self.length
        end=self.index_epoch
        return self.train_dataset[start:end,:],self.train_labels[start:end,:]

    def accuracy(self, predictions,labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])